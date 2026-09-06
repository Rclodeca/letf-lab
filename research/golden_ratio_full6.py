"""Full 6-asset Golden Ratio (matching the original r/LETFs post's SPMO +
DBMF diversifiers, not just variant C's 4-asset reduction), using the settled
signal (SPY 200SMA +/-1%, TIP 200SMA +/-0.1%) and SSO (not UPRO) as the
leverage sleeve. Compared against the 4-asset variant C over the identical
window, to see what adding momentum/managed-futures back actually costs or
buys versus the version all our tuning (band width, robustness grid, SSO/UPRO,
correlation-with-the-hold) was actually validated on.

DBMF's inception (2019-05-08) binds the window for the full 6-asset version,
same constraint as golden_ratio_delever.py's original variant A.
"""
import numpy as np
import pandas as pd

from ai_swing.data import get_price_service
from ai_swing.indicators import functions as F
from ai_swing.backtest.metrics import cagr, max_drawdown, sharpe, sortino, n_trades

ps = get_price_service()


def series(t):
    return ps.get_close_series(t)


def longest_dd_years(equity):
    eq = equity.dropna()
    peak_val, peak_date, max_gap_days = float("-inf"), eq.index[0], 0
    for dt, v in eq.items():
        if v >= peak_val:
            peak_val, peak_date = v, dt
        else:
            max_gap_days = max(max_gap_days, (dt - peak_date).days)
    return max_gap_days / 365.25


def portfolio_returns(rets, weights_on, weights_off, gate):
    pos = gate.shift(1).ffill().fillna(0.0)
    idx = rets.index.intersection(pos.index)
    rets, pos = rets.loc[idx], pos.loc[idx]
    on_ret = (rets[list(weights_on)] * pd.Series(weights_on)).sum(axis=1)
    off_ret = (rets[list(weights_off)] * pd.Series(weights_off)).sum(axis=1)
    return pos * on_ret + (1 - pos) * off_ret, pos


TICKERS = ["SPY", "TIP", "SSO", "SPMO", "VBR", "DBMF", "GLD", "TLT"]
inception = {t: series(t).dropna().index[0].date() for t in TICKERS}
print("Inception dates:")
for t in TICKERS:
    print(f"  {t:<6} {inception[t]}")

spy, tip = series("SPY"), series("TIP")
spy_gate = F.sma_gate(spy, 200, 0.01)
tip_gate = F.sma_gate(tip, 200, 0.001)
risk_on_gate = ((spy_gate == 1.0) & (tip_gate == 1.0)).astype(float)
risk_on_gate[spy_gate.isna() | tip_gate.isna()] = np.nan

VARIANTS = {
    "Full 6 (SSO+SPMO+VBR+DBMF+GLD+TLT)": {
        "on": {"SSO": 0.50, "SPMO": 0.10, "VBR": 0.10, "DBMF": 0.10, "GLD": 0.10, "TLT": 0.10},
        "off": {"SPMO": 0.20, "VBR": 0.20, "DBMF": 0.20, "GLD": 0.20, "TLT": 0.20},
    },
    "4-asset variant C (SSO+VBR+GLD+TLT)": {
        "on": {"SSO": 0.50, "VBR": 1 / 6, "GLD": 1 / 6, "TLT": 1 / 6},
        "off": {"VBR": 1 / 3, "GLD": 1 / 3, "TLT": 1 / 3},
    },
}

results = {}
common_start = str(series("DBMF").dropna().index[0].date())
print(f"\nCommon window (DBMF-bound, {common_start}..present):")
print(f"{'Variant':<38}{'CAGR':>8}{'MaxDD':>8}{'LDD(y)':>8}{'Sharpe':>8}{'Sortino':>8}{'Trades/yr':>11}")
for name, w in VARIANTS.items():
    assets = sorted(set(w["on"]) | set(w["off"]))
    px = pd.concat({t: series(t) for t in assets}, axis=1, sort=True).dropna()
    px = px.loc[common_start:]
    rets = px.pct_change().fillna(0.0)
    gate_aligned = risk_on_gate.reindex(px.index)
    strat_ret, pos = portfolio_returns(rets, w["on"], w["off"], gate_aligned)
    strat_ret = strat_ret.dropna()
    pos = pos.reindex(strat_ret.index)
    eq = (1 + strat_ret).cumprod()
    years = len(strat_ret) / 252
    print(f"{name:<38}{cagr(eq)*100:7.1f}%{max_drawdown(eq)*100:7.1f}%"
          f"{longest_dd_years(eq):8.1f}{sharpe(strat_ret):8.2f}{sortino(strat_ret):8.2f}"
          f"{n_trades(pos)/years:11.1f}")
    results[name] = (strat_ret, pos)

WIN = {"COVID 2020": ("2020-02-15", "2020-04-30"), "2022 bear": ("2022-01-01", "2022-12-31")}
print("\nCrash-window check:")
for name, (strat_ret, pos) in results.items():
    print(f"-- {name} --")
    for wname, (lo, hi) in WIN.items():
        rs = strat_ret.loc[lo:hi]
        if len(rs) < 5:
            continue
        es = (1 + rs).cumprod()
        dd = ((es / es.cummax() - 1).min()) * 100
        print(f"  {wname:<12} return {(es.iloc[-1]-1)*100:6.1f}%   maxDD {dd:6.1f}%")
