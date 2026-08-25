"""Golden Ratio de-lever with SSO (2x) instead of UPRO (3x) in the risk-on sleeve.

Bonus: SSO's inception (2006-06-21) is earlier than UPRO's (2009-06-25) and
earlier than TIP's (2003-12-05) isn't the binding constraint either -- so
swapping to SSO extends the real-data test back far enough to cover the 2008
GFC, which neither the UPRO variant nor anything else in this comparison so
far has been able to test on real (non-synthetic) data.
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


TICKERS = ["SPY", "TIP", "SSO", "UPRO", "VBR", "GLD", "TLT"]
inception = {t: series(t).dropna().index[0].date() for t in TICKERS}
print("Inception dates:")
for t in TICKERS:
    print(f"  {t:<6} {inception[t]}")

spy, tip = series("SPY"), series("TIP")
spy_gate = F.sma_gate(spy, period=200, threshold=0.005)
tip_gate = F.sma_gate(tip, period=200, threshold=0.001)
risk_on_gate = ((spy_gate == 1.0) & (tip_gate == 1.0)).astype(float)
risk_on_gate[spy_gate.isna() | tip_gate.isna()] = np.nan

OFF = {"VBR": 1 / 3, "GLD": 1 / 3, "TLT": 1 / 3}
VARIANTS = {
    "UPRO (3x) risk-on": {"UPRO": 0.50, "VBR": 1 / 6, "GLD": 1 / 6, "TLT": 1 / 6},
    "SSO (2x) risk-on": {"SSO": 0.50, "VBR": 1 / 6, "GLD": 1 / 6, "TLT": 1 / 6},
}


def portfolio_returns(rets, weights_on, weights_off, gate):
    pos = gate.shift(1).ffill().fillna(0.0)
    idx = rets.index.intersection(pos.index)
    rets, pos = rets.loc[idx], pos.loc[idx]
    on_ret = (rets[list(weights_on)] * pd.Series(weights_on)).sum(axis=1)
    off_ret = (rets[list(weights_off)] * pd.Series(weights_off)).sum(axis=1)
    return pos * on_ret + (1 - pos) * off_ret, pos


results = {}
print(f"\n{'Variant':<20}{'Window':<24}{'CAGR':>8}{'MaxDD':>8}{'LDD(y)':>8}{'Sharpe':>8}{'Sortino':>8}{'Trades/yr':>11}")
for name, on_w in VARIANTS.items():
    assets = sorted(set(on_w) | set(OFF))
    px = pd.concat({t: series(t) for t in assets}, axis=1, sort=True).dropna()
    rets = px.pct_change().fillna(0.0)
    gate_aligned = risk_on_gate.reindex(px.index)
    strat_ret, pos = portfolio_returns(rets, on_w, OFF, gate_aligned)
    strat_ret = strat_ret.dropna()
    pos = pos.reindex(strat_ret.index)
    eq = (1 + strat_ret).cumprod()
    years = len(strat_ret) / 252
    window = f"{eq.index[0].date()}..{eq.index[-1].date()}"
    print(f"{name:<20}{window:<24}{cagr(eq)*100:7.1f}%{max_drawdown(eq)*100:7.1f}%"
          f"{longest_dd_years(eq):8.1f}{sharpe(strat_ret):8.2f}{sortino(strat_ret):8.2f}"
          f"{n_trades(pos)/years:11.1f}")
    results[name] = (strat_ret, pos)

# Full crash-window set, now including 2008 GFC since SSO's window covers it.
WIN = {
    "GFC 2007-09": ("2007-10-01", "2009-06-30"),
    "COVID 2020": ("2020-02-15", "2020-04-30"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
}
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

# Direct apples-to-apples: trim SSO variant to UPRO's own window too.
upro_ret, _ = results["UPRO (3x) risk-on"]
sso_ret, sso_pos = results["SSO (2x) risk-on"]
common_start = upro_ret.index[0]
sso_trim = sso_ret.loc[common_start:]
sso_eq_trim = (1 + sso_trim).cumprod()
upro_eq = (1 + upro_ret).cumprod()
print(f"\nSame window as UPRO variant ({common_start.date()}..present):")
print(f"{'':<20}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}{'Sortino':>8}")
print(f"{'UPRO (3x)':<20}{cagr(upro_eq)*100:7.1f}%{max_drawdown(upro_eq)*100:7.1f}%"
      f"{sharpe(upro_ret):8.2f}{sortino(upro_ret):8.2f}")
print(f"{'SSO (2x)':<20}{cagr(sso_eq_trim)*100:7.1f}%{max_drawdown(sso_eq_trim)*100:7.1f}%"
      f"{sharpe(sso_trim):8.2f}{sortino(sso_trim):8.2f}")
