"""Daily LETF signal notifier.

Computes the SPY and QQQ vote-of-2 signals plus the standalone SMA-200 checks
defined in watchlist.py, diffs them against the previous run, and pushes a
short summary to Telegram.

Reuses the LETF Lab engine (`ai_swing.indicators.evaluator.evaluate_indicator`
and `ai_swing.data.PriceService`) so the numbers match the app exactly. No
database and no web server — indicators are evaluated from in-memory specs.

Run:
    python -m notify.daily_signals            # compute, send to Telegram, save state
    python -m notify.daily_signals --dry-run  # print the message only; no send, no save

Env:
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   (required unless --dry-run)
    PRICE_CACHE_DIR                        (parquet cache location; set by the workflow)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx

from ai_swing.data import get_price_service
from ai_swing.db.models import IndicatorType
from ai_swing.indicators.evaluator import evaluate_indicator

from notify.watchlist import SMA200_CHECKS, STRATEGIES

STATE_PATH = Path(__file__).parent / "state" / "last_signals.json"
CHECK = "✓"  # ✓
CROSS = "✗"  # ✗
MARK = {True: CHECK, False: CROSS}


def _latest(series):
    """Latest non-NaN (value, date) from a pandas Series, or (None, None)."""
    s = series.dropna()
    if s.empty:
        return None, None
    return float(s.iloc[-1]), s.index[-1].date()


def compute():
    """Return (signals, display, meta).

    signals : dict[str, bool]  — the monitored booleans, keyed for diffing.
    display : dict             — structured data for message formatting.
    meta    : dict[str, dict]  — per-key {label, kind} for rendering changes.
    """
    ps = get_price_service()

    # Prime + fetch each unique asset once. refresh() pulls recent bars (incl.
    # the latest close) and merges into the cache; get_close_series then returns
    # the up-to-date series. On a cold cache this self-heals to a full history.
    assets = {s["benchmark"] for s in STRATEGIES} | {c["asset"] for c in SMA200_CHECKS}
    prices_by_asset = {}
    for asset in sorted(assets):
        try:
            ps.refresh(asset, days=30)
        except Exception as exc:  # non-fatal: fall back to whatever is cached
            print(f"warn: refresh({asset}) failed: {exc}", file=sys.stderr)
        prices_by_asset[asset] = ps.get_close_series(asset)

    signals = {}
    meta = {}
    display = {"date": None, "strategies": [], "sma200": []}

    for spec in STRATEGIES:
        prices = prices_by_asset[spec["benchmark"]]
        returns = prices.pct_change()
        results = []
        for i, ind_spec in enumerate(spec["indicators"]):
            ind = SimpleNamespace(
                id=i,
                name=ind_spec["name"],
                type=IndicatorType(ind_spec["type"]),
                params=ind_spec["params"],
            )
            results.append(evaluate_indicator(ind, prices, returns=returns))
        score = sum(1 for r in results if r.gate_passed)
        risk_on = score >= spec["k"]
        key = f'{spec["name"]}_signal'
        signals[key] = risk_on
        meta[key] = {"label": f'{spec["name"]} signal', "kind": "verdict"}
        _, d = _latest(prices)
        if d and not display["date"]:
            display["date"] = d.isoformat()
        display["strategies"].append(
            {
                "name": spec["name"],
                "risk_on": risk_on,
                "score": score,
                "total": len(results),
                "gates": [(r.indicator_name, r.gate_passed) for r in results],
            }
        )

    for c in SMA200_CHECKS:
        prices = prices_by_asset[c["asset"]]
        sma200 = prices.rolling(window=200, min_periods=200).mean()
        p, _ = _latest(prices)
        s, _ = _latest(sma200)
        if p is None or s is None:
            fired = False
        elif c["op"] == "lt":
            fired = p < s * c["factor"]
        else:
            fired = p > s * c["factor"]
        signals[c["key"]] = fired
        meta[c["key"]] = {"label": c["label"], "kind": "flag"}
        display["sma200"].append({"label": c["label"], "fired": fired})

    return signals, display, meta


def load_prev_signals():
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text()).get("signals", {})
    except Exception:
        return {}


def diff_signals(prev, cur):
    """Return list of (key, old_bool, new_bool) for booleans that changed."""
    return [(k, prev[k], v) for k, v in cur.items() if k in prev and prev[k] != v]


def _render_change(kind, old, new):
    if kind == "verdict":
        word = {True: "RISK-ON", False: "risk-off"}
        return f"{word[old]} → {word[new]}"
    word = {True: "triggered", False: "cleared"}
    return f"{word[old]} → {word[new]}"


def format_message(display, changes, meta):
    lines = [f'\U0001f4ca LETF Lab — {display["date"] or date.today().isoformat()}', ""]
    for s in display["strategies"]:
        dot = "\U0001f7e2" if s["risk_on"] else "\U0001f534"
        state = "RISK-ON" if s["risk_on"] else "risk-off"
        gates = "  ".join(f"{name} {MARK[g]}" for name, g in s["gates"])
        lines.append(f'{s["name"]}  {dot} {state} ({s["score"]}/{s["total"]})')
        lines.append(f"   {gates}")
    lines.append("")
    lines.append("SMA-200 watch")
    for c in display["sma200"]:
        lines.append(f'  {MARK[c["fired"]]}  {c["label"]}')
    if changes:
        lines.append("")
        lines.append("⚠️ Changed today")
        for key, old, new in changes:
            info = meta.get(key, {"label": key, "kind": "flag"})
            lines.append(f'  {info["label"]}: {_render_change(info["kind"], old, new)}')
    return "\n".join(lines)


def send_telegram(text):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = httpx.post(url, json={"chat_id": chat_id, "text": text}, timeout=30)
    resp.raise_for_status()


def save_state(signals, display):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"date": display["date"], "signals": signals}
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the message to stdout; do not send to Telegram or save state",
    )
    args = ap.parse_args()

    signals, display, meta = compute()
    changes = diff_signals(load_prev_signals(), signals)
    message = format_message(display, changes, meta)

    if args.dry_run:
        print(message)
        return 0

    send_telegram(message)
    save_state(signals, display)
    print("Sent to Telegram and saved state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
