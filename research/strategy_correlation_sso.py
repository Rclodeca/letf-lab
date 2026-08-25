"""Re-run strategy_correlation.py's 2-strategy blend (Golden Ratio de-lever +
60/20/10/10 SSO/GLD/ZROZ/BIL hold), now using SSO (2x) instead of UPRO (3x) in
the Golden Ratio risk-on sleeve, per golden_ratio_sso_vs_upro.py's finding that
SSO gives up CAGR but comes out slightly ahead on Sharpe/Sortino over the same
window. Same common window as before (ZROZ-bound, Nov 2009-present) so this is
directly comparable to the UPRO-based numbers already discussed.
"""
import numpy as np
import pandas as pd

from ai_swing.data import get_price_service
from ai_swing.indicators import functions as F
from ai_swing.backtest.metrics import cagr, max_drawdown, sharpe, sortino

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


# --- Golden Ratio, SSO risk-on sleeve ---------------------------------------
spy, tip = series("SPY"), series("TIP")
spy_gate = F.sma_gate(spy, period=200, threshold=0.005)
tip_gate = F.sma_gate(tip, period=200, threshold=0.001)
risk_on_gate = ((spy_gate == 1.0) & (tip_gate == 1.0)).astype(float)
risk_on_gate[spy_gate.isna() | tip_gate.isna()] = np.nan

GR_ON = {"SSO": 0.50, "VBR": 1 / 6, "GLD": 1 / 6, "TLT": 1 / 6}
GR_OFF = {"VBR": 1 / 3, "GLD": 1 / 3, "TLT": 1 / 3}
gr_assets = sorted(set(GR_ON) | set(GR_OFF))
px = pd.concat({t: series(t) for t in gr_assets}, axis=1, sort=True).dropna()
rets = px.pct_change().fillna(0.0)
pos = risk_on_gate.reindex(px.index).shift(1).ffill().fillna(0.0)
gr_ret = (pos * (rets[list(GR_ON)] * pd.Series(GR_ON)).sum(axis=1)
          + (1 - pos) * (rets[list(GR_OFF)] * pd.Series(GR_OFF)).sum(axis=1)).dropna()

# --- SSO/GLD/ZROZ/BIL 60/20/10/10 quarterly hold ---------------------------
hold_eq, hold_ret = sim_quarterly({"SSO": 0.60, "GLD": 0.20, "ZROZ": 0.10, "BIL": 0.10})

# --- align to common window --------------------------------------------------
common = gr_ret.index.intersection(hold_ret.index)
gr_ret, hold_ret = gr_ret.loc[common], hold_ret.loc[common]
print(f"Common window: {common[0].date()}..{common[-1].date()} ({len(common)/252:.1f}y)\n")

corr = gr_ret.corr(hold_ret)
print(f"Full-period daily-return correlation: {corr:.3f}")

n_tail = max(1, int(len(common) * 0.05))
gr_worst_days = gr_ret.nsmallest(n_tail).index
hold_worst_days = hold_ret.nsmallest(n_tail).index
overlap = len(set(gr_worst_days) & set(hold_worst_days))
print(f"Overlap between the two worst-5%-day sets: {overlap}/{n_tail} days in common")

blend_ret = 0.5 * gr_ret + 0.5 * hold_ret
gr_eq = (1 + gr_ret).cumprod()
hold_eq_c = (1 + hold_ret).cumprod()
blend_eq = (1 + blend_ret).cumprod()

print(f"\n{'':<22}{'CAGR':>8}{'MaxDD':>8}{'LDD(y)':>8}{'Sharpe':>8}{'Sortino':>8}")
for name, eq, ret in [("Golden Ratio (SSO)", gr_eq, gr_ret),
                       ("SSO/GLD/ZROZ/BIL alone", hold_eq_c, hold_ret),
                       ("50/50 blend", blend_eq, blend_ret)]:
    print(f"{name:<22}{cagr(eq)*100:7.1f}%{max_drawdown(eq)*100:7.1f}%"
          f"{longest_dd_years(eq):8.1f}{sharpe(ret):8.2f}{sortino(ret):8.2f}")

WIN = {"COVID 2020": ("2020-02-15", "2020-04-30"), "2022 bear": ("2022-01-01", "2022-12-31")}
print("\nCrash-window check, blend vs. either alone:")
for name, ret in [("Golden Ratio (SSO)", gr_ret), ("SSO/GLD/ZROZ/BIL", hold_ret), ("50/50 blend", blend_ret)]:
    print(f"-- {name} --")
    for wname, (lo, hi) in WIN.items():
        rs = ret.loc[lo:hi]
        if len(rs) < 5:
            continue
        es = (1 + rs).cumprod()
        dd = ((es / es.cummax() - 1).min()) * 100
        print(f"  {wname:<12} return {(es.iloc[-1]-1)*100:6.1f}%   maxDD {dd:6.1f}%")

spy_ret_common = spy.pct_change().reindex(common).dropna()
idx2 = gr_ret.index.intersection(spy_ret_common.index)


def beta_r2_resid(y, x):
    y, x = y.loc[idx2], x.loc[idx2]
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)
    r2 = 1 - resid.var() / y.var()
    return beta, r2, resid


gr_beta, gr_r2, gr_resid = beta_r2_resid(gr_ret, spy_ret_common)
hold_beta, hold_r2, hold_resid = beta_r2_resid(hold_ret, spy_ret_common)
resid_corr = gr_resid.corr(hold_resid)

print(f"\nSPY-beta decomposition:")
print(f"  Golden Ratio (SSO): beta {gr_beta:.2f}   R^2 vs SPY {gr_r2*100:.0f}%")
print(f"  SSO/GLD/ZROZ/BIL:   beta {hold_beta:.2f}   R^2 vs SPY {hold_r2*100:.0f}%")
print(f"  Residual (SPY-beta stripped) correlation: {resid_corr:.3f}  (vs. {corr:.3f} raw)")
