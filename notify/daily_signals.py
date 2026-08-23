"""Daily LETF signal notifier.

Computes the SPY and QQQ vote-of-2 signals, two 3-state SMA-200 "traffic light"
strategies, and a raw-values panel — all defined in watchlist.py — diffs the
discrete states against the previous run, and pushes a summary to Telegram.

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

from notify.watchlist import RAW_ASSETS, STRATEGIES, TRAFFIC_LIGHTS

STATE_PATH = Path(__file__).parent / "state" / "last_signals.json"
CHECK, CROSS = "✓", "✗"
GATE_MARK = {True: CHECK, False: CROSS}
SIGNAL_DOT = {True: "🟢", False: "🔴"}
LIGHT_EMOJI = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}


def _latest(series):
    """Latest non-NaN value from a pandas Series, or None."""
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else None


def _latest_date(series):
    s = series.dropna()
    return s.index[-1].date() if not s.empty else None


def _sma(prices, period):
    return _latest(prices.rolling(window=period, min_periods=period).mean())


def _pct_day_change(prices):
    s = prices.dropna()
    if len(s) < 2:
        return None
    return (float(s.iloc[-1]) / float(s.iloc[-2]) - 1) * 100.0


def compute():
    """Return (signals, display, meta).

    signals : dict[str, bool|str]  — the discrete states, keyed for diffing.
    display : dict                 — structured data for message formatting.
    meta    : dict[str, dict]      — per-key {label, kind} for rendering changes.
    """
    ps = get_price_service()

    # Prime + fetch each unique asset once. refresh() pulls recent bars (incl.
    # the latest close) and merges into the cache; get_close_series then returns
    # the up-to-date series. On a cold cache this self-heals to a full history.
    assets = (
        {s["benchmark"] for s in STRATEGIES}
        | {t["asset"] for t in TRAFFIC_LIGHTS}
        | set(RAW_ASSETS)
    )
    prices_by_asset = {}
    for asset in sorted(assets):
        try:
            ps.refresh(asset, days=30)
        except Exception as exc:  # non-fatal: fall back to whatever is cached
            print(f"warn: refresh({asset}) failed: {exc}", file=sys.stderr)
        prices_by_asset[asset] = ps.get_close_series(asset)

    signals = {}
    meta = {}
    display = {"date": None, "strategies": [], "lights": [], "raw": []}

    # 1. Standard vote-of-k signals (SPY, QQQ).
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
        d = _latest_date(prices)
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

    # 2. 3-state SMA-200 traffic lights (SPY 200SMA, QQQ 200SMA).
    for t in TRAFFIC_LIGHTS:
        prices = prices_by_asset[t["asset"]]
        p = _latest(prices)
        s = _sma(prices, 200)
        if p is None or s is None:
            state = "HOLD"
        elif p > s * t["upper"]:
            state = "BUY"
        elif p < s * t["lower"]:
            state = "SELL"
        else:
            state = "HOLD"
        signals[t["key"]] = state
        meta[t["key"]] = {"label": t["name"], "kind": "light"}
        display["lights"].append({"name": t["name"], "state": state})

    # 3. Raw values panel.
    for asset in RAW_ASSETS:
        prices = prices_by_asset[asset]
        sma200 = _sma(prices, 200)
        display["raw"].append(
            {
                "asset": asset,
                "price": _latest(prices),
                "pct": _pct_day_change(prices),
                "sma250": _sma(prices, 250),
                "sma100": _sma(prices, 100),
                "sma200": sma200,
                "band_up": sma200 * 1.04 if sma200 is not None else None,
                "band_dn": sma200 * 0.97 if sma200 is not None else None,
            }
        )

    return signals, display, meta


def load_prev_signals():
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text()).get("signals", {})
    except Exception:
        return {}


def diff_signals(prev, cur):
    """Return list of (key, old, new) for discrete states that changed."""
    return [(k, prev[k], v) for k, v in cur.items() if k in prev and prev[k] != v]


def _render_change(kind, old, new):
    if kind == "verdict":
        word = {True: "RISK-ON", False: "risk-off"}
        return f"{word[old]} → {word[new]}"
    # light: states are already display strings ("BUY"/"HOLD"/"SELL")
    return f"{old} → {new}"


def _fmt(x):
    return f"{x:.2f}" if x is not None else "n/a"


def format_message(display, changes, meta):
    lines = [f'📊 LETF Lab — {display["date"] or date.today().isoformat()}', ""]

    for s in display["strategies"]:
        state = "RISK-ON" if s["risk_on"] else "risk-off"
        gates = "  ".join(f"{name} {GATE_MARK[g]}" for name, g in s["gates"])
        lines.append(f'{s["name"]}  {SIGNAL_DOT[s["risk_on"]]} {state} ({s["score"]}/{s["total"]})')
        lines.append(f"   {gates}")

    for lt in display["lights"]:
        lines.append(f'{lt["name"]}  {LIGHT_EMOJI[lt["state"]]} {lt["state"]}')

    lines.append("")
    lines.append("Raw values")
    for rv in display["raw"]:
        pct = f'{rv["pct"]:+.2f}%' if rv["pct"] is not None else "n/a"
        lines.append(f'{rv["asset"]}  {_fmt(rv["price"])}  ({pct})')
        lines.append(f'   SMA250 {_fmt(rv["sma250"])}   SMA100 {_fmt(rv["sma100"])}   SMA200 {_fmt(rv["sma200"])}')
        lines.append(f'   +4% {_fmt(rv["band_up"])}   −3% {_fmt(rv["band_dn"])}')

    if changes:
        lines.append("")
        lines.append("⚠️ Changed today")
        for key, old, new in changes:
            info = meta.get(key, {"label": key, "kind": "light"})
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
