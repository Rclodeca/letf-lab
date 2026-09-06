"""Quarterly calendar rebalance vs. drift-triggered rebalance (only rebalance
when a sleeve's actual weight breaches a band around its target, e.g. SSO
drifting from 60% target past 65%/55%) for the 60/20/10/10 SSO/GLD/ZROZ/BIL
hold. Lump-sum growth-of-a-dollar basis -- doesn't model DCA contributions
directly (see caveat printed below), but isolates the rebalance-TRIGGER
question: does waiting for a drift band vs. rebalancing every quarter
regardless change the outcome much?
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


def sim_quarterly(weights, start=None):
    px = pd.concat({t: series(t) for t in weights}, axis=1, sort=True).dropna()
    if start is not None:
        px = px.loc[start:]
    rets = px.pct_change().fillna(0.0)
    tickers = list(weights)
    sleeves = pd.Series({t: weights[t] for t in tickers}, dtype=float)
    prev = (px.index[0].year, (px.index[0].month - 1) // 3)
    eqv, n_rebal = [], 0
    for i, dt in enumerate(px.index):
        if i > 0:
            sleeves = sleeves * (1.0 + rets.loc[dt, tickers])
        k = (dt.year, (dt.month - 1) // 3)
        if k != prev:
            total = sleeves.sum()
            sleeves = pd.Series({t: total * weights[t] for t in tickers})
            prev, n_rebal = k, n_rebal + 1
        eqv.append(sleeves.sum())
    return pd.Series(eqv, index=px.index), n_rebal


def sim_band_triggered(weights, band_pts, start=None):
    """Rebalance only when any sleeve's actual weight drifts more than
    band_pts (absolute percentage points) from its target."""
    px = pd.concat({t: series(t) for t in weights}, axis=1, sort=True).dropna()
    if start is not None:
        px = px.loc[start:]
    rets = px.pct_change().fillna(0.0)
    tickers = list(weights)
    sleeves = pd.Series({t: weights[t] for t in tickers}, dtype=float)
    eqv, n_rebal = [], 0
    for i, dt in enumerate(px.index):
        if i > 0:
            sleeves = sleeves * (1.0 + rets.loc[dt, tickers])
        total = sleeves.sum()
        actual_w = sleeves / total
        drift = (actual_w - pd.Series(weights)).abs().max()
        if drift > band_pts:
            sleeves = pd.Series({t: total * weights[t] for t in tickers})
            n_rebal += 1
        eqv.append(sleeves.sum())
    return pd.Series(eqv, index=px.index), n_rebal


WEIGHTS = {"SSO": 0.60, "GLD": 0.20, "ZROZ": 0.10, "BIL": 0.10}
common_start = str(series("ZROZ").dropna().index[0].date())

print(f"Window: {common_start}..present\n")
print(f"{'Strategy':<28}{'CAGR':>8}{'MaxDD':>8}{'LDD(y)':>8}{'Sharpe':>8}{'Sortino':>8}{'Rebals':>8}")

eq_q, n_q = sim_quarterly(WEIGHTS, start=common_start)
print(f"{'Quarterly calendar':<28}{cagr(eq_q)*100:7.1f}%{max_drawdown(eq_q)*100:7.1f}%"
      f"{longest_dd_years(eq_q):8.1f}{sharpe(eq_q.pct_change().fillna(0.0)):8.2f}"
      f"{sortino(eq_q.pct_change().fillna(0.0)):8.2f}{n_q:8d}")

for band_pts in [0.05, 0.10, 0.15]:
    eq_b, n_b = sim_band_triggered(WEIGHTS, band_pts, start=common_start)
    ret_b = eq_b.pct_change().fillna(0.0)
    label = f"Band-triggered (±{band_pts*100:.0f}pp)"
    print(f"{label:<28}{cagr(eq_b)*100:7.1f}%{max_drawdown(eq_b)*100:7.1f}%"
          f"{longest_dd_years(eq_b):8.1f}{sharpe(ret_b):8.2f}{sortino(ret_b):8.2f}{n_b:8d}")

# Never rebalance at all, as the other extreme.
eq_n, n_n = sim_band_triggered(WEIGHTS, 1.0, start=common_start)  # band=100% never triggers
ret_n = eq_n.pct_change().fillna(0.0)
print(f"{'Never rebalance':<28}{cagr(eq_n)*100:7.1f}%{max_drawdown(eq_n)*100:7.1f}%"
      f"{longest_dd_years(eq_n):8.1f}{sharpe(ret_n):8.2f}{sortino(ret_n):8.2f}{n_n:8d}")

print("""
Caveat: this is lump-sum growth-of-a-dollar, not a DCA/contribution model.
DCA-only rebalancing (steering new contributions toward whatever's
underweight, no selling) is a real third option this script doesn't test
directly -- its effectiveness depends on contribution size relative to
portfolio size, which shrinks over time as the portfolio compounds. Early
on, when contributions are large relative to AUM, DCA alone can correct
drift about as well as a sell/buy rebalance. Later, once the portfolio is
large relative to what you're adding, DCA alone can't correct drift fast
enough and you need one of the two mechanisms tested above as a backstop.
""")
