"""Real-data test of the r/LETFs "1.7x Golden Ratio" strategy: SPY & TIP 200SMA
de-lever between a 50% UPRO / 50%-diversifiers risk-on mix and an all-diversifiers
risk-off mix (SPMO/VBR/DBMF/GLD/TLT).

The original post's "1988-Present" backtest necessarily splices in index/fund
proxies for most of its history -- UPRO (2009), SPMO (2015), and DBMF (2019) don't
exist that far back. This script runs the SAME allocation logic on 100% real ETF
prices (no synthetic reconstruction, same policy as backtest_holds.py), truncated
to whichever ticker's inception binds -- printed below -- to see how the claims
hold up on data that's actually real.

Three variants, each bound by a different ticker's inception:
  A. Full strategy (incl. DBMF)         -> bound by DBMF (~2019), ~6-7y
  B. DBMF dropped, weight -> other 4    -> bound by SPMO (~2015), ~10-11y
  C. DBMF + SPMO dropped, weight -> VBR/GLD/TLT -> bound by UPRO (2009), ~16-17y
     (directly comparable window to the SSO/GLD/ZROZ 60/20/20 hold in FINDINGS.md)

Signal: SPY 200SMA gate AND TIP 200SMA gate, both with a +/-0.5% hysteresis band
(matching the original post; F.sma_gate already implements this hysteresis).
Risk-on only when BOTH are on; risk-off otherwise. T+1 execution (shift by one
day), constant-mix daily rebalance within whichever state is active -- the
post's own trade-count claim (4/yr) implies low signal-state turnover, not
zero-rebalance drift, so this is the standard, defensible simplifying assumption
used elsewhere in this folder.
"""
import numpy as np
import pandas as pd

from ai_swing.data import get_price_service
from ai_swing.indicators import functions as F
from ai_swing.backtest.metrics import cagr, max_drawdown, sortino, sharpe, n_trades

ps = get_price_service()


def series(t):
    return ps.get_close_series(t)


TICKERS = ["SPY", "TIP", "UPRO", "SPMO", "VBR", "DBMF", "GLD", "TLT"]
inception = {t: series(t).dropna().index[0].date() for t in TICKERS}
print("Inception dates (binding constraint = latest date among the variant's tickers):")
for t in TICKERS:
    print(f"  {t:<6} {inception[t]}")

BAND_SPY = 0.005  # +/-0.5%, unchanged from the original post
BAND_TIP = 0.001  # +/-0.1%, per the post's follow-up feedback (TIP's own vol is
                  # much lower than SPY's, so 0.5% was an oversized band on it)

spy = series("SPY")
tip = series("TIP")
spy_gate = F.sma_gate(spy, period=200, threshold=BAND_SPY)
tip_gate = F.sma_gate(tip, period=200, threshold=BAND_TIP)
risk_on_gate = ((spy_gate == 1.0) & (tip_gate == 1.0)).astype(float)
risk_on_gate[spy_gate.isna() | tip_gate.isna()] = np.nan


def longest_dd_years(equity):
    """Longest stretch (in years) from an all-time-high to the next new
    all-time-high -- i.e. time spent underwater."""
    eq = equity.dropna()
    peak_val, peak_date, max_gap_days = -np.inf, eq.index[0], 0
    for dt, v in eq.items():
        if v >= peak_val:
            peak_val, peak_date = v, dt
        else:
            max_gap_days = max(max_gap_days, (dt - peak_date).days)
    return max_gap_days / 365.25


def portfolio_returns(rets, weights_on, weights_off, gate):
    """Constant-mix daily rebalance within whichever state is active, T+1 executed."""
    pos = gate.shift(1).ffill().fillna(0.0)
    idx = rets.index.intersection(pos.index)
    rets, pos = rets.loc[idx], pos.loc[idx]
    on_ret = (rets[list(weights_on)] * pd.Series(weights_on)).sum(axis=1)
    off_ret = (rets[list(weights_off)] * pd.Series(weights_off)).sum(axis=1)
    return pos * on_ret + (1 - pos) * off_ret, pos


VARIANTS = {
    "A. Full (incl. DBMF)": {
        "on": {"UPRO": 0.50, "SPMO": 0.10, "VBR": 0.10, "DBMF": 0.10, "GLD": 0.10, "TLT": 0.10},
        "off": {"SPMO": 0.20, "VBR": 0.20, "DBMF": 0.20, "GLD": 0.20, "TLT": 0.20},
    },
    "B. DBMF dropped": {
        "on": {"UPRO": 0.50, "SPMO": 0.125, "VBR": 0.125, "GLD": 0.125, "TLT": 0.125},
        "off": {"SPMO": 0.25, "VBR": 0.25, "GLD": 0.25, "TLT": 0.25},
    },
    "C. DBMF+SPMO dropped": {
        "on": {"UPRO": 0.50, "VBR": 1 / 6, "GLD": 1 / 6, "TLT": 1 / 6},
        "off": {"VBR": 1 / 3, "GLD": 1 / 3, "TLT": 1 / 3},
    },
}

print(f"\n{'Variant':<24}{'Window':<24}{'CAGR':>8}{'MaxDD':>8}{'LDD(y)':>8}{'Sharpe':>8}{'Sortino':>8}{'Trades/yr':>11}{'Time risk-on':>13}")
results = {}
for name, w in VARIANTS.items():
    assets = sorted(set(w["on"]) | set(w["off"]))
    px = pd.concat({t: series(t) for t in assets}, axis=1, sort=True).dropna()
    rets = px.pct_change().fillna(0.0)
    gate_aligned = risk_on_gate.reindex(px.index)
    strat_ret, pos = portfolio_returns(rets, w["on"], w["off"], gate_aligned)
    strat_ret = strat_ret.dropna()
    pos = pos.reindex(strat_ret.index)
    eq = (1 + strat_ret).cumprod()
    years = len(strat_ret) / 252
    window = f"{eq.index[0].date()}..{eq.index[-1].date()}"
    print(
        f"{name:<24}{window:<24}{cagr(eq)*100:7.1f}%{max_drawdown(eq)*100:7.1f}%"
        f"{longest_dd_years(eq):8.1f}{sharpe(strat_ret):8.2f}{sortino(strat_ret):8.2f}"
        f"{n_trades(pos)/years:11.1f}{pos.mean()*100:12.0f}%"
    )
    results[name] = (strat_ret, pos)

# SPY buy-hold benchmark over variant C's window (2009+, UPRO-bound) for reference.
spy_r = spy.pct_change().reindex(
    pd.concat({t: series(t) for t in ["UPRO", "VBR", "GLD", "TLT"]}, axis=1).dropna().index
).dropna()
spy_eq = (1 + spy_r).cumprod()
print(f"\n{'SPY buy-hold (variant C window)':<24}{'':<24}{cagr(spy_eq)*100:7.1f}%{max_drawdown(spy_eq)*100:7.1f}%"
      f"{longest_dd_years(spy_eq):8.1f}{sharpe(spy_r):8.2f}{sortino(spy_r):8.2f}")

# Did the de-lever actually fire in the two real crashes this window contains?
# (No dot-com, no GFC in any of these windows -- COVID 2020 and the 2022 bear
# are the only real stress tests available on 100% real data.)
WIN = {"COVID 2020": ("2020-02-15", "2020-04-30"), "2022 bear": ("2022-01-01", "2022-12-31")}
print("\nCrash-window check -- did risk-on/off actually flip, and did it help?")
for name, (strat_ret, pos) in results.items():
    print(f"-- {name} --")
    for wname, (lo, hi) in WIN.items():
        rs, pos_w = strat_ret.loc[lo:hi], pos.loc[lo:hi]
        if len(rs) < 5:
            continue
        es = (1 + rs).cumprod()
        dd = ((es / es.cummax() - 1).min()) * 100
        print(
            f"  {wname:<12} return {(es.iloc[-1]-1)*100:6.1f}%   maxDD {dd:6.1f}%   "
            f"time risk-on {pos_w.mean()*100:4.0f}%   flips {n_trades(pos_w)}"
        )

# --- vs. the existing SSO/GLD/ZROZ 60/20/20 quarterly hold (FINDINGS.md) --
# Same quarterly rebalance logic as backtest_holds.py, trimmed to the exact
# same start date as variant C for a true apples-to-apples window.
def sim_quarterly(weights, start=None):
    px = pd.concat({t: series(t) for t in weights}, axis=1, sort=True).dropna()
    if start is not None:
        px = px.loc[start:]
    rets = px.pct_change().fillna(0.0)
    tickers = list(weights)
    sleeves = pd.Series({t: weights[t] for t in tickers}, dtype=float)
    prev = (px.index[0].year, (px.index[0].month - 1) // 3)
    eqv = []
    for i, dt in enumerate(px.index):
        if i > 0:
            sleeves = sleeves * (1.0 + rets.loc[dt, tickers])
        k = (dt.year, (dt.month - 1) // 3)
        if k != prev:
            total = sleeves.sum()
            sleeves = pd.Series({t: total * weights[t] for t in tickers})
            prev = k
        eqv.append(sleeves.sum())
    eq = pd.Series(eqv, index=px.index)
    return eq, eq.pct_change().fillna(0.0)


c_strat_ret, c_pos = results["C. DBMF+SPMO dropped"]
common_start = c_strat_ret.index[0]
hold_eq, hold_ret = sim_quarterly({"SSO": 0.60, "GLD": 0.20, "ZROZ": 0.20}, start=common_start)
common_end = min(c_strat_ret.index[-1], hold_eq.index[-1])
c_eq_trim = (1 + c_strat_ret.loc[:common_end]).cumprod()
hold_eq_trim = hold_eq.loc[:common_end]
hold_ret_trim = hold_ret.loc[:common_end]

print(f"\nSame window ({hold_eq_trim.index[0].date()}..{hold_eq_trim.index[-1].date()}), "
      f"golden-ratio variant C vs. your existing SSO/GLD/ZROZ 60/20/20 quarterly hold:")
print(f"{'':<32}{'CAGR':>8}{'MaxDD':>8}{'LDD(y)':>8}{'Sharpe':>8}{'Sortino':>8}")
print(f"{'Golden Ratio (variant C)':<32}{cagr(c_eq_trim)*100:7.1f}%{max_drawdown(c_eq_trim)*100:7.1f}%"
      f"{longest_dd_years(c_eq_trim):8.1f}{sharpe(c_strat_ret.loc[:common_end]):8.2f}{sortino(c_strat_ret.loc[:common_end]):8.2f}")
print(f"{'SSO/GLD/ZROZ 60/20/20 (Q)':<32}{cagr(hold_eq_trim)*100:7.1f}%{max_drawdown(hold_eq_trim)*100:7.1f}%"
      f"{longest_dd_years(hold_eq_trim):8.1f}{sharpe(hold_ret_trim):8.2f}{sortino(hold_ret_trim):8.2f}")

for wname, (lo, hi) in WIN.items():
    rs = hold_ret_trim.loc[lo:hi]
    if len(rs) < 5:
        continue
    es = (1 + rs).cumprod()
    dd = ((es / es.cummax() - 1).min()) * 100
    print(f"  SSO/GLD/ZROZ {wname:<12} return {(es.iloc[-1]-1)*100:6.1f}%   maxDD {dd:6.1f}%")
