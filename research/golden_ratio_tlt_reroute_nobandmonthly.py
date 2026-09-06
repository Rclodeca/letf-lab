"""Golden Ratio full-6 + TLT-reroute (see golden_ratio_tlt_reroute.py), testing
two changes to the signal mechanics, isolated and combined:

  1. Drop all hysteresis bands -- SPY/TIP/TLT gates become plain SMA200
     zero-crossings (threshold=0) instead of the live +/-1%/+/-0.2%/+/-1%
     bands.
  2. Check monthly instead of daily -- gates are only sampled at month-end;
     whatever state that produces holds for the entire following month,
     rather than being re-evaluated (and able to flip) every trading day.

Four variants: baseline (bands+daily), no-bands-only, monthly-only, and both
combined (what was asked for), so it's clear how much each change
contributes on its own vs. together.

DBMF-bound window, same as golden_ratio_tlt_reroute.py.
"""
import numpy as np
import pandas as pd

from ai_swing.data import get_price_service
from ai_swing.indicators import functions as F
from ai_swing.backtest.metrics import cagr, max_drawdown, sortino, n_trades

ps = get_price_service()


def series(t):
    return ps.get_close_series(t)


TICKERS = ["SPY", "TIP", "TLT", "SSO", "SPMO", "VBR", "DBMF", "GLDM"]
common_start = str(series("DBMF").dropna().index[0].date())
px = pd.concat({t: series(t) for t in TICKERS}, axis=1, sort=True).dropna()
px = px.loc[common_start:]
rets = px.pct_change().fillna(0.0)

BASE_ON = {"SSO": 0.50, "SPMO": 0.10, "VBR": 0.10, "DBMF": 0.10, "GLDM": 0.10, "TLT": 0.10}
BASE_OFF = {"SPMO": 0.20, "VBR": 0.20, "DBMF": 0.20, "GLDM": 0.20, "TLT": 0.20}


def daily_weights(on_state, tlt_ok):
    base = dict(BASE_ON if on_state else BASE_OFF)
    if not tlt_ok:
        tlt_w = base.pop("TLT")
        base["DBMF"] = base.get("DBMF", 0.0) + tlt_w / 2
        base["GLDM"] = base.get("GLDM", 0.0) + tlt_w / 2
    return base


def monthly_hold(daily_series):
    """Sample at month-end, hold that value through the following month,
    T+1 execution (whole-series shift(1) fixes the month-end-day lookahead)."""
    monthly = daily_series.resample("ME").last()
    return monthly.reindex(daily_series.index, method="ffill")


def build_states(band_spy, band_tip, band_tlt, monthly: bool):
    spy_gate = F.sma_gate(px["SPY"], 200, band_spy)
    tip_gate = F.sma_gate(px["TIP"], 200, band_tip)
    tlt_gate = F.sma_gate(px["TLT"], 200, band_tlt)
    risk_on = ((spy_gate == 1.0) & (tip_gate == 1.0)).astype(float)
    risk_on[spy_gate.isna() | tip_gate.isna()] = np.nan

    if monthly:
        risk_on = monthly_hold(risk_on)
        tlt_gate = monthly_hold(tlt_gate)

    on = risk_on.shift(1).ffill().fillna(0.0)
    tlt_ok = tlt_gate.shift(1).ffill().fillna(1.0)
    return on, tlt_ok


def run(band_spy, band_tip, band_tlt, monthly: bool):
    on, tlt_ok = build_states(band_spy, band_tip, band_tlt, monthly)
    idx = rets.index.intersection(on.index).intersection(tlt_ok.index)
    on, tlt_ok, r = on.loc[idx], tlt_ok.loc[idx], rets.loc[idx]

    strat_ret = pd.Series(0.0, index=idx)
    for dt in idx:
        w = daily_weights(on.loc[dt] == 1.0, tlt_ok.loc[dt] == 1.0)
        strat_ret.loc[dt] = sum(w.get(t, 0.0) * r.loc[dt, t] for t in TICKERS if t in w)

    eq = (1 + strat_ret).cumprod()
    pos = on * 2 + tlt_ok
    return strat_ret, pos, eq


VARIANTS = {
    "Baseline (bands, daily)": dict(band_spy=0.01, band_tip=0.002, band_tlt=0.01, monthly=False),
    "No bands, daily": dict(band_spy=0.0, band_tip=0.0, band_tlt=0.0, monthly=False),
    "Bands, monthly check": dict(band_spy=0.01, band_tip=0.002, band_tlt=0.01, monthly=True),
    "No bands, monthly check": dict(band_spy=0.0, band_tip=0.0, band_tlt=0.0, monthly=True),
}

print(f"Window: {px.index[0].date()}..{px.index[-1].date()}  (DBMF-bound, full-6 mix + TLT reroute)\n")

results = {}
print(f"{'Variant':<28}{'CAGR':>8}{'MaxDD':>8}{'Sortino':>9}{'Trades/yr':>11}")
for label, kw in VARIANTS.items():
    strat_ret, pos, eq = run(**kw)
    years = len(strat_ret) / 252
    results[label] = (strat_ret, pos, eq)
    print(f"{label:<28}{cagr(eq)*100:7.1f}%{max_drawdown(eq)*100:7.1f}%{sortino(strat_ret):9.2f}{n_trades(pos)/years:11.1f}")

WIN = {"COVID 2020": ("2020-02-15", "2020-04-30"), "2022 bear": ("2022-01-01", "2022-12-31")}
print("\nCrash-window check:")
for label in VARIANTS:
    strat_ret, pos, eq = results[label]
    print(f"-- {label} --")
    for wname, (lo, hi) in WIN.items():
        rs = strat_ret.loc[lo:hi]
        if len(rs) < 5:
            continue
        es = (1 + rs).cumprod()
        dd = ((es / es.cummax() - 1).min()) * 100
        print(f"  {wname:<12} return {(es.iloc[-1]-1)*100:6.1f}%   maxDD {dd:6.1f}%")
