"""Overfitting check: wiggle SPY SMA period, TIP SMA period, SPY band, and TIP
band each a little around the chosen settings (200/200-day SMAs, 1%/0.1% bands)
and see whether performance holds up across the neighborhood, or whether the
chosen point is an isolated lucky peak.

Small perturbations only (not a wide exploratory grid) -- that's the actual
overfitting question: is the chosen combo sitting on a stable plateau, or does
it fall off a cliff a few days/bps away in any direction?
"""
import itertools

import numpy as np
import pandas as pd

from ai_swing.data import get_price_service
from ai_swing.indicators import functions as F
from ai_swing.backtest.metrics import cagr, max_drawdown, sortino, n_trades

ps = get_price_service()


def series(t):
    return ps.get_close_series(t)


spy, tip = series("SPY"), series("TIP")
GR_ON = {"SSO": 0.50, "VBR": 1 / 6, "GLD": 1 / 6, "TLT": 1 / 6}
GR_OFF = {"VBR": 1 / 3, "GLD": 1 / 3, "TLT": 1 / 3}
assets = sorted(set(GR_ON) | set(GR_OFF))
px = pd.concat({t: series(t) for t in assets}, axis=1, sort=True).dropna()
rets = px.pct_change().fillna(0.0)

SPY_SMA = [190, 195, 200, 205, 210]
TIP_SMA = [190, 195, 200, 205, 210]
SPY_BAND = [0.005, 0.0075, 0.01, 0.0125, 0.015]
TIP_BAND = [0.0005, 0.00075, 0.001, 0.00125, 0.0015]
CHOSEN = (200, 200, 0.01, 0.001)

rows = []
for spy_sma, tip_sma, spy_band, tip_band in itertools.product(SPY_SMA, TIP_SMA, SPY_BAND, TIP_BAND):
    spy_gate = F.sma_gate(spy, spy_sma, spy_band)
    tip_gate = F.sma_gate(tip, tip_sma, tip_band)
    gate = ((spy_gate == 1.0) & (tip_gate == 1.0)).astype(float)
    gate[spy_gate.isna() | tip_gate.isna()] = np.nan
    pos = gate.reindex(px.index).shift(1).ffill().fillna(0.0)
    on_ret = (rets[list(GR_ON)] * pd.Series(GR_ON)).sum(axis=1)
    off_ret = (rets[list(GR_OFF)] * pd.Series(GR_OFF)).sum(axis=1)
    strat_ret = (pos * on_ret + (1 - pos) * off_ret).dropna()
    pos_a = pos.reindex(strat_ret.index)
    eq = (1 + strat_ret).cumprod()
    yrs = len(strat_ret) / 252
    rows.append({
        "spy_sma": spy_sma, "tip_sma": tip_sma, "spy_band": spy_band, "tip_band": tip_band,
        "cagr": cagr(eq), "max_dd": max_drawdown(eq), "sortino": sortino(strat_ret),
        "trades_yr": n_trades(pos_a) / yrs,
    })

df = pd.DataFrame(rows)
n = len(df)
chosen_row = df[(df.spy_sma == CHOSEN[0]) & (df.tip_sma == CHOSEN[1])
                & (df.spy_band == CHOSEN[2]) & (df.tip_band == CHOSEN[3])].iloc[0]

print(f"Grid: {n} combinations "
      f"(SPY SMA {SPY_SMA}, TIP SMA {TIP_SMA}, SPY band {[f'{b*100:.2f}%' for b in SPY_BAND]}, "
      f"TIP band {[f'{b*100:.3f}%' for b in TIP_BAND]})\n")

print(f"Chosen point (200/200, 1%/0.1%): CAGR {chosen_row.cagr*100:.1f}%  MaxDD {chosen_row.max_dd*100:.1f}%  "
      f"Sortino {chosen_row.sortino:.2f}  trades/yr {chosen_row.trades_yr:.1f}\n")

for col, fmt in [("cagr", "%"), ("max_dd", "%"), ("sortino", ""), ("trades_yr", "")]:
    vals = df[col] * 100 if fmt == "%" else df[col]
    chosen_val = chosen_row[col] * 100 if fmt == "%" else chosen_row[col]
    pct_rank = (df[col] <= chosen_row[col]).mean() * 100
    print(f"{col:<10} min {vals.min():7.2f}{fmt}  p5 {vals.quantile(.05):7.2f}{fmt}  "
          f"median {vals.median():7.2f}{fmt}  p95 {vals.quantile(.95):7.2f}{fmt}  "
          f"max {vals.max():7.2f}{fmt}   |  chosen={chosen_val:.2f}{fmt} (percentile {pct_rank:.0f})")

print(f"\nCombos with Sortino < 1.2: {(df.sortino < 1.2).sum()}/{n} ({(df.sortino < 1.2).mean()*100:.0f}%)")
print(f"Combos with Sortino < 1.0: {(df.sortino < 1.0).sum()}/{n} ({(df.sortino < 1.0).mean()*100:.0f}%)")

print("\nWorst 5 combos by Sortino:")
worst = df.nsmallest(5, "sortino")
for _, r in worst.iterrows():
    print(f"  SPY_sma={r.spy_sma} TIP_sma={r.tip_sma} SPY_band={r.spy_band*100:.2f}% TIP_band={r.tip_band*100:.3f}%  "
          f"CAGR {r.cagr*100:.1f}%  MaxDD {r.max_dd*100:.1f}%  Sortino {r.sortino:.2f}")

print("\nBest 5 combos by Sortino:")
best = df.nlargest(5, "sortino")
for _, r in best.iterrows():
    print(f"  SPY_sma={r.spy_sma} TIP_sma={r.tip_sma} SPY_band={r.spy_band*100:.2f}% TIP_band={r.tip_band*100:.3f}%  "
          f"CAGR {r.cagr*100:.1f}%  MaxDD {r.max_dd*100:.1f}%  Sortino {r.sortino:.2f}")
