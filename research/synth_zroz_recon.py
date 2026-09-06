"""Reconstruct a synthetic ZROZ (25-27y zero-coupon Treasury strips) back to
1988 (^SP500TR's earliest Yahoo data; ^NDX/^TYX go back further to
1985-10/1977-02, so 1988 is the binding constraint, ~2y short of the
r/LETFs post's 1986 "real data" start) from the 30y CMT yield (^TYX), and
re-run the post's Tier 3 winner (QLD, vote-of-2 on QQQ own-underlying with
+/-5% SMA buffer, off -> ZROZ) against it, to see how close we get to their
reported gross Sortino 1.325 / edge +0.367 without any cash/TLT substitute.

Zero-coupon bond return model (duration + convexity, fit by regression
against REAL ZROZ over the 2009-11..2026 overlap, same
calibrate-then-trust approach as synth_calibrate.py for the LETFs):

    r_t = carry_t + a*dy_t + b*dy_t^2
    carry_t = y_{t-1}/252            (running yield accrual)
    dy_t    = y_t - y_{t-1}          (daily change in 30y yield, decimal)
    a ~= -modified_duration, b ~= 0.5*duration*(duration+1)  (convexity)

a, b are fit by OLS on the overlap, then applied to the full 1977-2026
^TYX history to build the synthetic series.
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

D = {t: hist(t) for t in ["^SP500TR", "^NDX", "^IRX", "^TYX", "ZROZ"]}
sp, ndx, tyx, real_zroz = D["^SP500TR"], D["^NDX"], D["^TYX"], D["ZROZ"]
irx = D["^IRX"] / 100.0
def rate(idx): return irx.reindex(idx).ffill().fillna(0.02)

# --- fit duration + convexity against real ZROZ over the overlap ----------
y = (tyx / 100.0).reindex(pd.date_range(tyx.index[0], tyx.index[-1])).ffill()
dy = y.diff()
carry = y.shift(1) / 252

real_zroz_ret = real_zroz.pct_change()
overlap = pd.concat({"real": real_zroz_ret, "dy": dy.reindex(real_zroz_ret.index),
                      "carry": carry.reindex(real_zroz_ret.index)}, axis=1).dropna()
target = overlap["real"] - overlap["carry"]
X = np.column_stack([overlap["dy"].to_numpy(), (overlap["dy"] ** 2).to_numpy()])
coef, *_ = np.linalg.lstsq(X, target.to_numpy(), rcond=None)
a, b = coef
implied_duration = -a
print(f"fit: a={a:.3f} (implied modified duration {implied_duration:.1f}y)  b={b:.3f}  "
      f"(vs {implied_duration*(implied_duration+1)/2:.1f} expected convexity coef for a pure zero of that duration)")

synth_zroz_ret = (carry + a * dy + b * dy ** 2).rename("synth_ZROZ")

# calibration check over the real-ZROZ overlap
j = pd.concat([synth_zroz_ret, real_zroz_ret], axis=1, join="inner").dropna()
sr = (1 + j.iloc[:, 0]).cumprod(); rr = (1 + j.iloc[:, 1]).cumprod()
yrs = len(j) / 252
corr = j.iloc[:, 0].corr(j.iloc[:, 1])
print(f"\ncalibration vs real ZROZ, {j.index[0].date()}..{j.index[-1].date()} ({yrs:.1f}y): "
      f"daily-return corr {corr:.3f}  CAGR synth {(sr.iloc[-1]**(1/yrs)-1)*100:.1f}% vs real "
      f"{(rr.iloc[-1]**(1/yrs)-1)*100:.1f}%  maxDD synth {max_drawdown(sr)*100:.0f}% vs real {max_drawdown(rr)*100:.0f}%")

# --- rebuild QLD/TQQQ synthetics (same model as elsewhere in this folder) -
SPREAD = 0.005
NDX_DIV = 0.007
EXP = {"QLD": 0.0095, "TQQQ": 0.0084}
LEV = {"QLD": 2, "TQQQ": 3}

def synth_ret(name):
    r = ndx.pct_change() + NDX_DIV / 252
    L, ter = LEV[name], EXP[name]
    fin = rate(r.index) + SPREAD
    return (L * r - (L - 1) * (fin / 252) - ter / 252).rename(name)

# own-underlying k>=2, +/-5% SMA buffer (matches the r/LETFs Tier 3 winner)
ndxr = ndx.pct_change()
k2_buf5 = F.vote_of_k([F.sma_gate(ndx, 250, 0.05), F.sma_gate(ndx, 100, 0.05),
                        F.realized_vol_gate(ndxr, 21, 0.40), F.ar1_gate(ndxr, 30, 0.0)], 2)

# daily T+1 execution (what we've been doing everywhere in this folder)
pos_daily = k2_buf5.shift(1).ffill().fillna(0.0)

# T+1 MONTH-END execution: the gate is still computed daily (SMA250/100 etc.
# need daily granularity to mean anything), but the position only updates
# once a month -- take the gate's value on each month's last trading day,
# then hold that decision for the entire following month (applied starting
# the next trading day, i.e. T+1 from month-end).
month_end_gate = k2_buf5.resample("ME").last()
pos_monthend = month_end_gate.reindex(k2_buf5.index, method="ffill").shift(1).ffill().fillna(0.0)

EXEC = {"daily": pos_daily, "month-end": pos_monthend}

START = "1988-01-04"  # earliest ^SP500TR date on Yahoo; ~2y short of the post's 1986 start
sp_r = sp.pct_change().loc[START:].dropna()
sp_eq = (1 + sp_r).cumprod()

WIN = {"DOT-COM 00-02": ("2000-03-01", "2002-12-31"), "GFC 07-09": ("2007-10-01", "2009-06-30"),
       "COVID 2020": ("2020-02-15", "2020-04-30"), "2022 bear": ("2022-01-01", "2022-12-31")}

strat_returns = {}
for inst in ["QLD", "TQQQ"]:
    inst_ret = synth_ret(inst)
    for exec_name, pos in EXEC.items():
        idx = inst_ret.index.intersection(synth_zroz_ret.index)
        strat_ret = (pos.reindex(idx).fillna(0) * inst_ret.reindex(idx)
                     + (1 - pos.reindex(idx).fillna(0)) * synth_zroz_ret.reindex(idx).fillna(0))
        r = strat_ret.loc[START:].dropna()
        eq = (1 + r).cumprod()
        p = pos.reindex(r.index).fillna(0)
        print(f"\n=== {inst}, vote-of-2 own-signal (+/-5% buf), {exec_name} execution, off -> synthetic ZROZ, "
              f"{r.index[0].date()}..{r.index[-1].date()} ({len(r)/252:.1f}y) ===")
        print(f"CAGR {cagr(eq)*100:6.1f}%   Max DD {max_drawdown(eq)*100:6.1f}%   Sortino {sortino(r):6.2f}   "
              f"flips {n_trades(p)}   time-in {p.mean()*100:.0f}%")
        print(f"Sortino edge vs SPY (same window, SPY Sortino {sortino(sp_r):.2f}): {sortino(r)-sortino(sp_r):+.3f}")
        if inst == "QLD":
            strat_returns[exec_name] = strat_ret

print("\n=== crash-window return / max drawdown, QLD variant, daily vs month-end execution ===")
for exec_name, strat_ret in strat_returns.items():
    print(f"-- {exec_name} --")
    for wname, (lo, hi) in WIN.items():
        rs = strat_ret.loc[lo:hi].dropna()
        if len(rs) < 5: continue
        es = (1 + rs).cumprod()
        print(f"{wname:<16} return {(es.iloc[-1]-1)*100:6.0f}%   maxDD {((es/es.cummax()-1).min())*100:6.0f}%")
