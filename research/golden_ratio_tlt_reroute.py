"""Golden Ratio (full 6: SSO+SPMO+VBR+DBMF+GLDM+TLT) with a THIRD, independent
gate on TLT itself: SMA200 +/-1% (same band style as the SPY gate). When TLT
is below its own band, TLT's slot (10% risk-on / 20% risk-off, per the live
full-6 weights) is rerouted 50/50 into DBMF and GLDM instead of held as TLT.

Rationale: TLT is a fixed diversifier present in BOTH risk-on and risk-off
states of the SPY+TIP gate -- that gate never actually reduces TLT exposure,
it only controls the SSO/leverage sleeve. This adds a dedicated escape hatch
for the one regime (2022-style, rate/inflation-driven) where TLT itself is
the thing losing money, using DBMF/GLDM -- the two instruments that actually
worked in that regime on real 2022 data (see prior conversation: DBMF +20.5%,
GLDM/GLD +0.8%, vs. TLT -29.4%).

DBMF's inception (2019-05-08) binds the window, same constraint as
golden_ratio_full6.py.
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

spy_gate = F.sma_gate(px["SPY"], 200, 0.01)
tip_gate = F.sma_gate(px["TIP"], 200, 0.002)
risk_on = ((spy_gate == 1.0) & (tip_gate == 1.0)).astype(float)
risk_on[spy_gate.isna() | tip_gate.isna()] = np.nan

tlt_gate = F.sma_gate(px["TLT"], 200, 0.01)  # new, independent TLT-health gate

BASE_ON = {"SSO": 0.50, "SPMO": 0.10, "VBR": 0.10, "DBMF": 0.10, "GLDM": 0.10, "TLT": 0.10}
BASE_OFF = {"SPMO": 0.20, "VBR": 0.20, "DBMF": 0.20, "GLDM": 0.20, "TLT": 0.20}


def daily_weights(on_state, tlt_ok):
    """Return the weight dict for a single day given (risk_on bool, tlt_healthy bool)."""
    base = dict(BASE_ON if on_state else BASE_OFF)
    if not tlt_ok:
        tlt_w = base.pop("TLT")
        base["DBMF"] = base.get("DBMF", 0.0) + tlt_w / 2
        base["GLDM"] = base.get("GLDM", 0.0) + tlt_w / 2
    else:
        base.setdefault("TLT", base.get("TLT", 0.0))
    return base


def run(use_tlt_reroute: bool):
    on = risk_on.shift(1).ffill().fillna(0.0)  # T+1 execution
    tlt_ok = tlt_gate.shift(1).ffill().fillna(1.0)  # default "healthy" during warmup
    idx = rets.index.intersection(on.index).intersection(tlt_ok.index)
    on, tlt_ok, r = on.loc[idx], tlt_ok.loc[idx], rets.loc[idx]

    strat_ret = pd.Series(0.0, index=idx)
    for dt in idx:
        w = daily_weights(on.loc[dt] == 1.0, (tlt_ok.loc[dt] == 1.0) if use_tlt_reroute else True)
        strat_ret.loc[dt] = sum(w.get(t, 0.0) * r.loc[dt, t] for t in TICKERS if t in w)

    eq = (1 + strat_ret).cumprod()
    # "position" for trade counting = combined on/off + tlt-health state
    pos = on * 2 + (tlt_ok if use_tlt_reroute else 1.0)
    return strat_ret, pos, eq


print(f"Window: {px.index[0].date()}..{px.index[-1].date()}  (DBMF-bound, full-6 mix)\n")

results = {}
for label, flag in [("Baseline full-6 (static TLT)", False), ("+ TLT SMA200 1% reroute to DBMF/GLDM", True)]:
    strat_ret, pos, eq = run(flag)
    years = len(strat_ret) / 252
    results[label] = (strat_ret, pos, eq)
    print(f"{label:<40}{'CAGR':>8}{'MaxDD':>8}{'Sortino':>9}{'Trades/yr':>11}")
    print(f"{'':<40}{cagr(eq)*100:7.1f}%{max_drawdown(eq)*100:7.1f}%{sortino(strat_ret):9.2f}{n_trades(pos)/years:11.1f}")

WIN = {"COVID 2020": ("2020-02-15", "2020-04-30"), "2022 bear": ("2022-01-01", "2022-12-31")}
print("\nCrash-window check:")
for label in results:
    strat_ret, pos, eq = results[label]
    print(f"-- {label} --")
    for wname, (lo, hi) in WIN.items():
        rs = strat_ret.loc[lo:hi]
        if len(rs) < 5:
            continue
        es = (1 + rs).cumprod()
        dd = ((es / es.cummax() - 1).min()) * 100
        print(f"  {wname:<12} return {(es.iloc[-1]-1)*100:6.1f}%   maxDD {dd:6.1f}%")

# How often, and for how long, does the reroute actually fire?
tlt_ok_shifted = tlt_gate.shift(1).ffill().reindex(results["+ TLT SMA200 1% reroute to DBMF/GLDM"][2].index)
print(f"\nTLT-unhealthy (rerouted) time: {(tlt_ok_shifted == 0.0).mean()*100:.0f}% of days, "
      f"{n_trades(tlt_ok_shifted)} flips over {len(tlt_ok_shifted)/252:.1f} years")
