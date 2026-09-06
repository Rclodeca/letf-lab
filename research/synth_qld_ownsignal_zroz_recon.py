"""Reconciliation check vs. the r/LETFs "40-year LETF rotation backtest"
post's Tier 3 winner: qld_voteK2_sma250_100_vol21_40_ar30_off_zroz.

Their config, reproduced as closely as data allows:
  - Signal computed on the QQQ/NDX underlying itself (NOT SPY, which is what
    our earlier 8-way sweep used for all 4 instruments -- that's the main
    reason those numbers didn't match this post).
  - Vote-of-2 across: price > SMA250(NDX)*1.05, price > SMA100(NDX)*1.05,
    realized_vol_21d(NDX) < 40%, AR(1)_30d(NDX) > 0.
  - On-state: 100% QLD (2x NDX, synthetic, same reconstruction as elsewhere
    in this folder). Off-state: ZROZ.
  - ZROZ real history only goes back to 2009-11; TLT (20+y Treasury, real
    ETF) goes back to 2002-07-30 and is the proxy already used elsewhere in
    FINDINGS.md ("Hold 2004-2026, ZROZ proxied by TLT") for the same reason.
    Window here is therefore 2002-07-30..2026-08-24 (~24y) -- misses the
    2000-02 dot-com crash, but covers GFC/COVID/2022.
  - Gross (no tax modeling) -- the post reports net-of-tax numbers under two
    tax models (annual-netting and per-swing), which this script does not
    attempt to reproduce; treat this as a gross-vs-gross comparison only.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from ai_swing.indicators import functions as F
from ai_swing.backtest.metrics import cagr, max_drawdown, sortino, n_trades

def hist(t):
    df = yf.Ticker(t).history(period="max", auto_adjust=True)
    if df is None or df.empty: return pd.Series(dtype=float, name=t)
    s = df["Close"].copy(); s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index().rename(t)

D = {t: hist(t) for t in ["^SP500TR", "^NDX", "^IRX", "TLT"]}
sp, ndx, tlt = D["^SP500TR"], D["^NDX"], D["TLT"]
irx = D["^IRX"] / 100.0
def rate(idx): return irx.reindex(idx).ffill().fillna(0.02)

SPREAD = 0.005
QLD_TER = 0.0095
NDX_DIV = 0.007

def synth_qld():
    r = ndx.pct_change() + NDX_DIV / 252
    fin = rate(r.index) + SPREAD
    return (2 * r - 1 * (fin / 252) - QLD_TER / 252).rename("QLD 2x")

qld_ret = synth_qld()
tlt_ret = tlt.pct_change().rename("TLT (ZROZ proxy)")
ndxr = ndx.pct_change()

k2_ownsignal = F.vote_of_k([
    F.sma_gate(ndx, 250, 0.05), F.sma_gate(ndx, 100, 0.05),
    F.realized_vol_gate(ndxr, 21, 0.40), F.ar1_gate(ndxr, 30, 0.0),
], 2)

pos = k2_ownsignal.shift(1).ffill().fillna(0.0)
idx = qld_ret.index.intersection(tlt_ret.index)
strat_ret = (pos.reindex(idx).fillna(0) * qld_ret.reindex(idx)
             + (1 - pos.reindex(idx).fillna(0)) * tlt_ret.reindex(idx).fillna(0))

START = str(tlt.index[0].date())  # 2002-07-30, TLT-limited
r = strat_ret.loc[START:].dropna()
eq = (1 + r).cumprod()
p = pos.reindex(r.index).fillna(0)

print(f"=== Reconciliation: QLD, vote-of-2 on QQQ/NDX (5% buf on SMA250/100), off->TLT(~ZROZ) ===")
print(f"Window: {r.index[0].date()}..{r.index[-1].date()} ({len(r)/252:.1f}y, TLT-inception-limited)\n")
print(f"CAGR       {cagr(eq)*100:6.1f}%")
print(f"Max DD     {max_drawdown(eq)*100:6.1f}%")
print(f"Sortino    {sortino(r):6.2f}")
print(f"# flips    {n_trades(p)}")
print(f"Time in QLD: {p.mean()*100:.0f}%")

sp_r = sp.pct_change().loc[START:].dropna(); sp_eq = (1 + sp_r).cumprod()
print(f"\nSPY 1x buy-hold (same window): CAGR {cagr(sp_eq)*100:6.1f}%  Max DD {max_drawdown(sp_eq)*100:6.1f}%  "
      f"Sortino {sortino(sp_r):6.2f}")
print(f"Sortino edge vs SPY: {sortino(r) - sortino(sp_r):+.3f}")

WIN = {"GFC 2007-2009": ("2007-10-01", "2009-06-30"), "COVID 2020": ("2020-02-15", "2020-04-30"),
       "2022 bear": ("2022-01-01", "2022-12-31")}
print("\n=== crash-window return / max drawdown ===")
for wname, (lo, hi) in WIN.items():
    rs = strat_ret.loc[lo:hi].dropna()
    if len(rs) < 5: continue
    es = (1 + rs).cumprod()
    print(f"{wname:<16} return {(es.iloc[-1]-1)*100:6.0f}%   maxDD {((es/es.cummax()-1).min())*100:6.0f}%")

# also: same signal/instrument but risk-off = cash, for a same-window apples-to-apples
cash_ret = (rate(sp.index) / 252)
strat_cash = (pos.reindex(idx).fillna(0) * qld_ret.reindex(idx)
              + (1 - pos.reindex(idx).fillna(0)) * cash_ret.reindex(idx).fillna(0))
rc = strat_cash.loc[START:].dropna(); ec = (1 + rc).cumprod()
print(f"\n(for comparison, same signal/instrument, risk-off=CASH instead of TLT/ZROZ:)")
print(f"CAGR {cagr(ec)*100:6.1f}%  Max DD {max_drawdown(ec)*100:6.1f}%  Sortino {sortino(rc):6.2f}")
