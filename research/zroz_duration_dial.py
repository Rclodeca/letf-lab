"""How much of the SSO/GLD/ZROZ hold's behavior is specifically ZROZ's extreme
duration (~27-28y, near its maturity since it's zero-coupon)? Dials duration down
two ways and re-runs the same quarterly-rebalance hold from golden_ratio_delever.py /
backtest_holds.py, same window (ZROZ-bound, Nov 2009-present) for apples-to-apples:

  1. 60/20/20 ZROZ            -- baseline (FINDINGS.md)
  2. 60/20/20 TLT             -- full swap to less-extreme duration (~17-18y)
  3. 60/20/10/10 ZROZ+TLT     -- half the ZROZ sleeve, half into TLT (partial dial-down)
  4. 60/20/10/10 ZROZ+BIL     -- half the ZROZ sleeve, half into cash (pure de-risk,
                                 no duration exposure on that half at all)

Crash-window check (COVID 2020, 2022 bear) on each shows the tradeoff directly:
how much COVID-style hedge you give up per unit of 2022-style pain you shed.
"""
import pandas as pd

from ai_swing.data import get_price_service
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


def sim_quarterly(weights, start=None, end=None):
    px = pd.concat({t: series(t) for t in weights}, axis=1, sort=True).dropna()
    if start is not None:
        px = px.loc[start:]
    if end is not None:
        px = px.loc[:end]
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


TICKERS = ["SSO", "GLD", "ZROZ", "TLT", "BIL"]
inception = {t: series(t).dropna().index[0].date() for t in TICKERS}
print("Inception dates:")
for t in TICKERS:
    print(f"  {t:<6} {inception[t]}")

VARIANTS = {
    "60/20/20 ZROZ (baseline)": {"SSO": 0.60, "GLD": 0.20, "ZROZ": 0.20},
    "60/20/20 TLT (full swap)": {"SSO": 0.60, "GLD": 0.20, "TLT": 0.20},
    "60/20/10/10 ZROZ+TLT": {"SSO": 0.60, "GLD": 0.20, "ZROZ": 0.10, "TLT": 0.10},
    "60/20/10/10 ZROZ+BIL": {"SSO": 0.60, "GLD": 0.20, "ZROZ": 0.10, "BIL": 0.10},
}

# Common window: bound by ZROZ's own inception, so every variant is compared
# on identical dates even though TLT/BIL alone could run further back.
common_start = str(inception["ZROZ"])

results = {}
print(f"\nCommon window (ZROZ-bound, {common_start}..present):")
print(f"{'Variant':<28}{'CAGR':>8}{'MaxDD':>8}{'LDD(y)':>8}{'Sharpe':>8}{'Sortino':>8}")
for name, w in VARIANTS.items():
    eq, ret = sim_quarterly(w, start=common_start)
    results[name] = (eq, ret)
    print(f"{name:<28}{cagr(eq)*100:7.1f}%{max_drawdown(eq)*100:7.1f}%"
          f"{longest_dd_years(eq):8.1f}{sharpe(ret):8.2f}{sortino(ret):8.2f}")

WIN = {"COVID 2020": ("2020-02-15", "2020-04-30"), "2022 bear": ("2022-01-01", "2022-12-31")}
print("\nCrash-window check:")
for name, (eq, ret) in results.items():
    print(f"-- {name} --")
    for wname, (lo, hi) in WIN.items():
        rs = ret.loc[lo:hi]
        if len(rs) < 5:
            continue
        es = (1 + rs).cumprod()
        dd = ((es / es.cummax() - 1).min()) * 100
        print(f"  {wname:<12} return {(es.iloc[-1]-1)*100:6.1f}%   maxDD {dd:6.1f}%")
