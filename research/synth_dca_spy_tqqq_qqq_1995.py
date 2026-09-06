"""200SMA SPY -> TQQQ with a DCA-to-QQQ risk-off ramp and a QQQ-euphoria
valve, starting 1995.

Rules (as specified):
  1. BUY:  SPY crosses above SMA200(SPY)*1.04  -> 100% TQQQ.
  2. SELL: SPY crosses below SMA200(SPY)*0.97  -> 100% cash, then DCA
     linearly into QQQ (1x NDX) over the next 10 months (210 trading days).
     If the BUY trigger (rule 1) fires again before the ramp finishes, abort
     the ramp immediately and go 100% TQQQ. If the ramp completes with no
     BUY trigger, stay 100% QQQ until either a BUY trigger (-> TQQQ) fires.
  3. EUPHORIA VALVE: whenever QQQ (NDX) trades >30% above its own
     SMA200(NDX), force 100% cash regardless of the state above (protects
     against holding 3x/1x NDX exposure into a dot-com-style blow-off top).
     Position reverts to whatever rule 1/2 dictates once QQQ falls back
     under the 30% threshold.

Ambiguities resolved (flagging in case a different reading was intended):
  - "10 months" = 210 trading days (21/mo), ramp is linear in trading days.
  - The euphoria valve overrides ALL exposure (TQQQ and the QQQ ramp/hold),
    not just the TQQQ leg.
  - Signals use day-t closes; positions are applied lagged by 1 day (same
    no-lookahead convention as the other scripts in this folder).

TQQQ/QQQ are synthetic, reconstructed from ^NDX (see synth_calibrate.py for
calibration vs the real ETFs: daily-return corr 0.995-0.999 over overlap).
Cash is the ^IRX short-rate proxy (only de-risk asset covering back to
1995 daily; BIL starts 2007, SGOV starts 2020).
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

def synth_ret(L, ter):
    r = ndx.pct_change() + NDX_DIV / 252
    fin = rate(r.index) + SPREAD
    return L * r - (L - 1) * (fin / 252) - ter / 252

tqqq_ret = synth_ret(3, TER_TQQQ).rename("TQQQ 3x")
qqq_ret = (ndx.pct_change() + NDX_DIV / 252).rename("QQQ 1x")
cash_ret = (rate(sp.index) / 252).rename("cash")

BUY, SELL, EUPHORIA = 0.04, 0.03, 0.30
RAMP_DAYS = 210  # 10 months * 21 trading days

idx = sp.index.intersection(ndx.index)
sma_sp = sp.rolling(200, min_periods=200).mean().reindex(idx)
sma_ndx = ndx.rolling(200, min_periods=200).mean().reindex(idx)
sp_i, ndx_i = sp.reindex(idx), ndx.reindex(idx)

w_tqqq = np.zeros(len(idx)); w_qqq = np.zeros(len(idx)); w_cash = np.zeros(len(idx))
hyst = np.nan   # 1 = risk-on (TQQQ side), 0 = risk-off (DCA/QQQ side)
dca_start = None
euphoria_days, buy_events, sell_events = [], [], []

for i in range(len(idx)):
    if np.isnan(sma_sp.iloc[i]) or np.isnan(sma_ndx.iloc[i]):
        w_cash[i] = 1.0
        continue
    spy_above = sp_i.iloc[i] > sma_sp.iloc[i] * (1 + BUY)
    spy_below = sp_i.iloc[i] < sma_sp.iloc[i] * (1 - SELL)
    if np.isnan(hyst):
        hyst = 1.0 if sp_i.iloc[i] > sma_sp.iloc[i] else 0.0
    else:
        if hyst == 0.0 and spy_above:
            hyst = 1.0; dca_start = None; buy_events.append(idx[i])
        elif hyst == 1.0 and spy_below:
            hyst = 0.0; dca_start = i; sell_events.append(idx[i])

    if hyst == 1.0:
        wt, wq, wc = 1.0, 0.0, 0.0
    else:
        elapsed = i - dca_start if dca_start is not None else RAMP_DAYS
        frac = min(1.0, max(0.0, elapsed / RAMP_DAYS))
        wt, wq, wc = 0.0, frac, 1 - frac

    if ndx_i.iloc[i] > sma_ndx.iloc[i] * (1 + EUPHORIA):
        wt, wq, wc = 0.0, 0.0, 1.0
        euphoria_days.append(idx[i])

    w_tqqq[i], w_qqq[i], w_cash[i] = wt, wq, wc

W = pd.DataFrame({"tqqq": w_tqqq, "qqq": w_qqq, "cash": w_cash}, index=idx).shift(1).fillna(0.0)

strat_ret = (W["tqqq"] * tqqq_ret.reindex(idx).fillna(0)
             + W["qqq"] * qqq_ret.reindex(idx).fillna(0)
             + W["cash"] * cash_ret.reindex(idx).fillna(0))

START = "1995-01-01"
r = strat_ret.loc[START:].dropna()
eq = (1 + r).cumprod()
Wr = W.loc[r.index]
pos_binary = (Wr["tqqq"] > 0).astype(float)  # for n_trades: count TQQQ entries/exits

print(f"=== SPY 200SMA(+4%/-3%) -> TQQQ, DCA-to-QQQ over 10mo, QQQ euphoria valve @ +30%, "
      f"{r.index[0].date()}..{r.index[-1].date()} ({len(r)/252:.1f}y) ===")
print(f"CAGR       {cagr(eq)*100:6.1f}%")
print(f"Max DD     {max_drawdown(eq)*100:6.1f}%")
print(f"Sortino    {sortino(r):6.2f}")
print(f"TQQQ entries/exits: {n_trades(pos_binary)}   sell(DCA-start) events: {len(sell_events)}   "
      f"euphoria-override days: {len(euphoria_days)}")
print(f"Time allocated: TQQQ {Wr['tqqq'].mean()*100:.0f}%  QQQ(ramp) {Wr['qqq'].mean()*100:.0f}%  cash {Wr['cash'].mean()*100:.0f}%")
if euphoria_days:
    print(f"Euphoria-valve windows touched: {euphoria_days[0].date()} .. {euphoria_days[-1].date()} "
          f"(first/last of {len(euphoria_days)} flagged days)")

for name, rr in [("TQQQ 3x buy-hold", tqqq_ret.loc[START:].dropna()),
                  ("S&P 1x buy-hold", sp.pct_change().loc[START:].dropna())]:
    e = (1 + rr).cumprod()
    print(f"\n{name}: CAGR {cagr(e)*100:6.1f}%  Max DD {max_drawdown(e)*100:6.1f}%  Sortino {sortino(rr):6.2f}")

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
