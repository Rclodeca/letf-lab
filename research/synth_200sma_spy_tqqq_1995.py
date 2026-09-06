"""200SMA SPY -> TQQQ, starting 1995, de-risking to CASH (risk-free proxy).

Trigger: SPY vs its 200-day SMA, ASYMMETRIC band — buy (go long TQQQ) when
SPY > SMA*1.04, sell (move to cash) when SPY < SMA*0.97. Inside the band the
previous state holds (hysteresis). `ai_swing.indicators.functions.sma_gate`
only supports a symmetric ±threshold, so the gate is reimplemented locally
below with independent buy/sell thresholds.

Risk-off leg is CASH: BIL only starts 2007-05 and SGOV only starts 2020-06,
neither covers back to 1995, so the short-rate proxy (^IRX) already used
elsewhere in this repo's synthetic backtests is the only de-risk asset with
full 1995-2026 coverage. TQQQ is reconstructed synthetically from ^NDX (see
synth_calibrate.py for the calibration: daily-return corr 0.995-0.999 vs the
real ETF over their overlap).
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

SPREAD = 0.005          # borrow spread above risk-free (matches synth_scenarios.py)
TER_TQQQ = 0.0084
NDX_DIV = 0.007         # NDX is price-return; crude dividend add for total-return basis

def synth_tqqq():
    r = ndx.pct_change() + NDX_DIV / 252
    fin = rate(r.index) + SPREAD
    return (3 * r - 2 * (fin / 252) - TER_TQQQ / 252).rename("TQQQ 3x")

def asym_sma_gate(prices, period, buy_thresh, sell_thresh):
    """Hysteresis gate with independent buy/sell thresholds around SMA(period).

    Risk-on only above SMA*(1+buy_thresh); risk-off only below
    SMA*(1-sell_thresh); holds previous state inside the band. NaN warmup.
    """
    sma = prices.rolling(window=period, min_periods=period).mean()
    upper = sma * (1.0 + buy_thresh)
    lower = sma * (1.0 - sell_thresh)
    p, u, l, s = prices.to_numpy(), upper.to_numpy(), lower.to_numpy(), sma.to_numpy()
    out = np.full(len(prices), np.nan)
    state = np.nan
    for i in range(len(prices)):
        if np.isnan(s[i]) or np.isnan(p[i]):
            out[i] = np.nan; state = np.nan; continue
        if np.isnan(state):
            state = 1.0 if p[i] > s[i] else 0.0
        if p[i] > u[i]: state = 1.0
        elif p[i] < l[i]: state = 0.0
        out[i] = state
    return pd.Series(out, index=prices.index)

tqqq_ret = synth_tqqq()
cash_ret = (rate(sp.index) / 252).rename("cash")

BUY, SELL = 0.04, 0.03  # buy at SMA+4%, sell at SMA-3%
sig = asym_sma_gate(sp, 200, BUY, SELL)
pos = sig.shift(1).ffill().fillna(0.0)

idx = tqqq_ret.index
strat_ret = (pos.reindex(idx).fillna(0) * tqqq_ret
             + (1 - pos.reindex(idx).fillna(0)) * cash_ret.reindex(idx).fillna(0))

START = "1995-01-01"
r = strat_ret.loc[START:].dropna()
eq = (1 + r).cumprod()
p = pos.reindex(r.index).fillna(0)

print(f"=== 200SMA SPY -> TQQQ (buy +{BUY*100:.0f}% / sell -{SELL*100:.0f}%, risk-off: CASH), "
      f"{r.index[0].date()}..{r.index[-1].date()} ({len(r)/252:.1f}y) ===")
print(f"CAGR       {cagr(eq)*100:6.1f}%")
print(f"Max DD     {max_drawdown(eq)*100:6.1f}%")
print(f"Sortino    {sortino(r):6.2f}")
print(f"# flips    {n_trades(p)}")
print(f"Time in TQQQ: {p.mean()*100:.0f}%")

# reference: buy-hold TQQQ and SPY 1x over the same window
for name, rr in [("TQQQ 3x buy-hold", tqqq_ret.loc[START:].dropna()),
                  ("S&P 1x buy-hold", sp.pct_change().loc[START:].dropna())]:
    e = (1 + rr).cumprod()
    print(f"\n{name}: CAGR {cagr(e)*100:6.1f}%  Max DD {max_drawdown(e)*100:6.1f}%  Sortino {sortino(rr):6.2f}")

# crash-window breakdown
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
