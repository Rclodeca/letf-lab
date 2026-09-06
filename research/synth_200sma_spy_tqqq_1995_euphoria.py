"""Strat 1 (200SMA SPY -> TQQQ, sell straight to CASH) + the QQQ-euphoria
valve from strat 2, WITHOUT the DCA-to-QQQ ramp. Isolates whether the
euphoria valve alone helps, since the DCA ramp was shown to hurt (it
re-risked into QQQ on a fixed calendar during still-falling markets in
2000-02 and 2007-09).

Rules: buy TQQQ when SPY > SMA200(SPY)*1.04; sell to 100% cash when
SPY < SMA200(SPY)*0.97 (hysteresis, prior state holds in between); AND
force 100% cash whenever QQQ (^NDX) > SMA200(NDX)*1.30, overriding the
above (protects against holding TQQQ into a dot-com-style blow-off top).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from ai_swing.backtest.metrics import cagr, max_drawdown, sortino, n_trades

def hist(t):
    df = yf.Ticker(t).history(period="max", auto_adjust=True)
    if df is None or df.empty: return pd.Series(dtype=float, name=t)
    s = df["Close"].copy(); s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index().rename(t)

D = {t: hist(t) for t in ["^SP500TR", "^NDX", "^IRX"]}
sp, ndx = D["^SP500TR"], D["^NDX"]
irx = D["^IRX"] / 100.0
def rate(idx): return irx.reindex(idx).ffill().fillna(0.02)

SPREAD = 0.005
TER_TQQQ = 0.0084
NDX_DIV = 0.007

def synth_tqqq():
    r = ndx.pct_change() + NDX_DIV / 252
    fin = rate(r.index) + SPREAD
    return (3 * r - 2 * (fin / 252) - TER_TQQQ / 252).rename("TQQQ 3x")

BUY, SELL, EUPHORIA = 0.04, 0.03, 0.30

idx = sp.index.intersection(ndx.index)
sma_sp = sp.rolling(200, min_periods=200).mean().reindex(idx)
sma_ndx = ndx.rolling(200, min_periods=200).mean().reindex(idx)
sp_i, ndx_i = sp.reindex(idx), ndx.reindex(idx)

hyst_arr = np.full(len(idx), np.nan)
euphoria_days = []
hyst = np.nan
for i in range(len(idx)):
    if np.isnan(sma_sp.iloc[i]) or np.isnan(sma_ndx.iloc[i]):
        continue
    spy_above = sp_i.iloc[i] > sma_sp.iloc[i] * (1 + BUY)
    spy_below = sp_i.iloc[i] < sma_sp.iloc[i] * (1 - SELL)
    if np.isnan(hyst):
        hyst = 1.0 if sp_i.iloc[i] > sma_sp.iloc[i] else 0.0
    else:
        if hyst == 0.0 and spy_above: hyst = 1.0
        elif hyst == 1.0 and spy_below: hyst = 0.0
    w = hyst
    if ndx_i.iloc[i] > sma_ndx.iloc[i] * (1 + EUPHORIA):
        w = 0.0
        euphoria_days.append(idx[i])
    hyst_arr[i] = w

sig = pd.Series(hyst_arr, index=idx)
pos = sig.shift(1).ffill().fillna(0.0)

tqqq_ret = synth_tqqq()
cash_ret = (rate(sp.index) / 252).rename("cash")

strat_ret = (pos.reindex(idx).fillna(0) * tqqq_ret.reindex(idx)
             + (1 - pos.reindex(idx).fillna(0)) * cash_ret.reindex(idx).fillna(0))

START = "1995-01-01"
r = strat_ret.loc[START:].dropna()
eq = (1 + r).cumprod()
p = pos.reindex(r.index).fillna(0)

print(f"=== strat 1 + euphoria valve: SPY 200SMA(+4%/-3%) -> TQQQ/CASH, force cash if QQQ >30% above its 200SMA, "
      f"{r.index[0].date()}..{r.index[-1].date()} ({len(r)/252:.1f}y) ===")
print(f"CAGR       {cagr(eq)*100:6.1f}%")
print(f"Max DD     {max_drawdown(eq)*100:6.1f}%")
print(f"Sortino    {sortino(r):6.2f}")
print(f"# flips    {n_trades(p)}")
print(f"Time in TQQQ: {p.mean()*100:.0f}%   euphoria-override days: {len(euphoria_days)}")

WIN = {
    "DOT-COM 2000-2002": ("2000-03-01", "2002-12-31"),
    "GFC 2007-2009": ("2007-10-01", "2009-06-30"),
    "COVID 2020": ("2020-02-15", "2020-04-30"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
}
print("\n=== crash-window return / max drawdown ===")
print(f"{'window':<20}{'strategy':>12}{'strat maxDD':>14}{'TQQQ b&h':>12}{'TQQQ maxDD':>13}")
for wname, (lo, hi) in WIN.items():
    rs = strat_ret.loc[lo:hi].dropna(); rt = tqqq_ret.loc[lo:hi].dropna()
    if len(rs) < 5: continue
    es = (1 + rs).cumprod(); et = (1 + rt).cumprod()
    dds = (es / es.cummax() - 1).min(); ddt = (et / et.cummax() - 1).min()
    print(f"{wname:<20}{(es.iloc[-1]-1)*100:>11.0f}%{dds*100:>13.0f}%{(et.iloc[-1]-1)*100:>11.0f}%{ddt*100:>12.0f}%")
