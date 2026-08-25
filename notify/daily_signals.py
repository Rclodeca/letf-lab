"""Daily LETF signal notifier.

Computes the SPY and QQQ vote-of-2 signals, the Golden Ratio SPY+TIP
de-lever gate, two 3-state SMA-200 "traffic light" strategies, and a
raw-values panel — all defined in watchlist.py — diffs the discrete states
against the previous run, and pushes a summary to Telegram.

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
from ai_swing.indicators import functions as F
from ai_swing.indicators.evaluator import evaluate_indicator

from notify.watchlist import DUAL_GATES, RAW_ASSETS, STRATEGIES, TRAFFIC_LIGHTS

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
        | {i["asset"] for g in DUAL_GATES for i in g["indicators"]}
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
    display = {"date": None, "strategies": [], "lights": [], "raw": [], "dual_gate_raw": []}

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

    # 1b. AND-combined multi-asset gates (e.g. the Golden Ratio SPY+TIP
    # de-lever signal) — risk-on only when every indicator passes.
    for spec in DUAL_GATES:
        results = []
        for ind_spec in spec["indicators"]:
            prices = prices_by_asset[ind_spec["asset"]]
            returns = prices.pct_change()
            ind = SimpleNamespace(
                id=0,
                name=ind_spec["name"],
                type=IndicatorType(ind_spec["type"]),
                params=ind_spec["params"],
            )
            results.append(evaluate_indicator(ind, prices, returns=returns))
        score = sum(1 for r in results if r.gate_passed)
        risk_on = score == len(results)
        key = spec["key"]
        signals[key] = risk_on
        meta[key] = {"label": spec["name"], "kind": "verdict"}
        display["strategies"].append(
            {
                "name": spec["name"],
                "risk_on": risk_on,
                "score": score,
                "total": len(results),
                "gates": [(r.indicator_name, r.gate_passed) for r in results],
            }
        )
        # Raw band values, per indicator's own SMA period + hysteresis threshold
        # (e.g. SPY ±0.5%, TIP ±0.1% — distinct from the fixed traffic-light bands).
        for ind_spec in spec["indicators"]:
            asset = ind_spec["asset"]
            period = int(ind_spec["params"].get("period", 200))
            threshold = float(ind_spec["params"].get("threshold", 0.0))
            prices = prices_by_asset[asset]
            sma = _sma(prices, period)
            display["dual_gate_raw"].append(
                {
                    "gate": spec["name"],
                    "asset": asset,
                    "period": period,
                    "threshold": threshold,
                    "price": _latest(prices),
                    "sma": sma,
                    "band_up": sma * (1 + threshold) if sma is not None else None,
                    "band_dn": sma * (1 - threshold) if sma is not None else None,
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
        returns = prices.pct_change()
        sma250 = _sma(prices, 250)
        sma100 = _sma(prices, 100)
        sma200 = _sma(prices, 200)
        display["raw"].append(
            {
                "asset": asset,
                "price": _latest(prices),
                "pct": _pct_day_change(prices),
                "sma250": sma250,
                "sma250_up": sma250 * 1.05 if sma250 is not None else None,
                "sma250_dn": sma250 * 0.95 if sma250 is not None else None,
                "sma100": sma100,
                "sma100_up": sma100 * 1.05 if sma100 is not None else None,
                "sma100_dn": sma100 * 0.95 if sma100 is not None else None,
                "sma200": sma200,
                "band_up": sma200 * 1.04 if sma200 is not None else None,
                "band_dn": sma200 * 0.97 if sma200 is not None else None,
                "vol21": _latest(F.realized_vol(returns, window=21)),
                "ar1": _latest(F.ar1_coefficient(returns, window=30)),
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


def _banner_lines(changes, meta):
    """Actionable-change banner lines for the top of the message.

    Fires for: vote-of-2 signal flips (either direction — risk-on and risk-off
    are both actionable) and 200SMA light changes only when the NEW state is
    BUY or SELL. A change *to* HOLD is intentionally ignored (no action needed).
    """
    out = []
    for key, old, new in changes:
        info = meta.get(key, {})
        kind = info.get("kind")
        if kind == "verdict":
            arrow = f"{SIGNAL_DOT[old]} → {SIGNAL_DOT[new]}"
            name = info.get("label", key).replace(" signal", "")
            word = "RISK-ON" if new else "RISK-OFF"
            out.append(f"{arrow}  {name} now {word}")
        elif kind == "light" and new in ("BUY", "SELL"):
            arrow = f"{LIGHT_EMOJI[old]} → {LIGHT_EMOJI[new]}"
            out.append(f'{arrow}  {info.get("label", key)} now {new}')
    return out


def _ladder(price, levels):
    """Render a mini-ladder: the group's levels plus the price, sorted high→low,
    with the price row marked. `levels` is a list of (label, value)."""
    rows = [(price, "price", True)] + [(v, lbl, False) for lbl, v in levels]
    rows = [r for r in rows if r[0] is not None]
    rows.sort(key=lambda r: r[0], reverse=True)
    out = []
    for value, label, is_price in rows:
        marker = "▶" if is_price else " "
        out.append(f"  {marker} {value:>8.2f}  {label}")
    return out


def format_message(display, changes, meta):
    lines = []

    # Big attention banner at the very top on actionable changes. Being first,
    # it also becomes the phone's notification preview.
    banner = _banner_lines(changes, meta)
    if banner:
        lines.append("<b>🚨 SIGNAL CHANGE</b>")
        lines += banner
        lines.append("")

    lines.append(f'📊 LETF Lab — {display["date"] or date.today().isoformat()}')
    lines.append("")

    for s in display["strategies"]:
        state = "RISK-ON" if s["risk_on"] else "risk-off"
        gates = "  ".join(f"{name} {GATE_MARK[g]}" for name, g in s["gates"])
        lines.append(f'{s["name"]}  {SIGNAL_DOT[s["risk_on"]]} {state} ({s["score"]}/{s["total"]})')
        lines.append(f"   {gates}")

    for lt in display["lights"]:
        lines.append(f'{lt["name"]}  {LIGHT_EMOJI[lt["state"]]} {lt["state"]}')

    # Raw values panel — monospace (<pre>) so the ladders align. Two groups per
    # asset: the vote-of-2 trend SMAs, and the 200SMA traffic-light bands.
    lines.append("")
    lines.append("Raw values")
    block = []
    for rv in display["raw"]:
        pct = f'{rv["pct"]:+.2f}%' if rv["pct"] is not None else "n/a"
        block.append(f'{rv["asset"]}   ({pct})')
        block.append("  vote-of-2 SMAs (±5% band)")
        block += _ladder(
            rv["price"],
            [
                ("SMA100 +5%", rv["sma100_up"]), ("SMA100", rv["sma100"]), ("SMA100 -5%", rv["sma100_dn"]),
                ("SMA250 +5%", rv["sma250_up"]), ("SMA250", rv["sma250"]), ("SMA250 -5%", rv["sma250_dn"]),
            ],
        )
        block.append("  vote-of-2 vol / mom")
        vol = f'{rv["vol21"] * 100:.2f}%' if rv["vol21"] is not None else "n/a"
        ar1 = f'{rv["ar1"]:+.3f}' if rv["ar1"] is not None else "n/a"
        block.append(f"    {vol:>8}  Vol21d   cap 40%")
        block.append(f"    {ar1:>8}  AR(1)    min 0")
        block.append("  200SMA bands")
        block += _ladder(
            rv["price"],
            [("+4% band", rv["band_up"]), ("SMA200", rv["sma200"]), ("−3% band", rv["band_dn"])],
        )
        block.append("")

    # Golden Ratio-style dual-gate raw bands — each indicator's own SMA
    # period + hysteresis threshold (e.g. SPY ±0.5%, TIP ±0.1%).
    if display["dual_gate_raw"]:
        by_gate = {}
        for rv in display["dual_gate_raw"]:
            by_gate.setdefault(rv["gate"], []).append(rv)
        for gate_name, rows in by_gate.items():
            block.append(gate_name)
            for rv in rows:
                pct = f'{rv["threshold"] * 100:.1f}%'
                block.append(f'  {rv["asset"]}  SMA{rv["period"]} (±{pct} band)')
                block += _ladder(
                    rv["price"],
                    [
                        (f"+{pct} band", rv["band_up"]),
                        (f'SMA{rv["period"]}', rv["sma"]),
                        (f"-{pct} band", rv["band_dn"]),
                    ],
                )
            block.append("")
    lines.append("<pre>" + "\n".join(block).rstrip() + "</pre>")

    return "\n".join(lines)


def send_telegram(text):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = httpx.post(
        url,
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
        timeout=30,
    )
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
