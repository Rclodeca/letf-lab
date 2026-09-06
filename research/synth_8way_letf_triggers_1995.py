"""Sweep: {SSO, UPRO, QLD, TQQQ} x {k>=2, k>=2 (+/-5% buf), SPY
200SMA(+4%/-3%)}, all triggered off SPY (not each instrument's own
underlying), de-risking to CASH, extended back to 1995 via synthetic
reconstruction.

- SSO = 2x SPY, UPRO = 3x SPY, QLD = 2x QQQ/NDX, TQQQ = 3x QQQ/NDX, all
  reconstructed from ^SP500TR/^NDX via the daily-reset model calibrated in
  synth_calibrate.py (corr 0.995-0.999 vs real ETFs over their overlap).
- k>=2: vote-of-2 across SMA250(SPY)/SMA100(SPY)/Vol21d(SPY)<40%/AR1(SPY)>0,
  strict 0% band on the two SMA gates (same "k2" signal as
  synth_scenarios.py's "k>=2 SPY->SSO", and how the app's seeded SMA250/
  SMA100 indicators are actually defined in backend/scripts/seed.py).
- k>=2 (+/-5% buf): same vote-of-2, but SMA250/SMA100 each get a symmetric
  +/-5% hysteresis band instead of the strict 0% band. Not how the app's
  seeded indicators are configured today — added here to check whether a
  5% buffer (as opposed to the 200SMA strategy's separate, unrelated 3.5%
  buffer) would reduce k>=2's whipsaw/turnover.
- SPY 200SMA(+4%/-3%): asymmetric hysteresis band, same as
  synth_200sma_spy_tqqq_1995.py's strat 1 (buy above SMA*1.04, sell below
  SMA*0.97, hold state in between).
Both signals are computed once on SPY and applied to all four instruments
(i.e. this asks "does SPY's trend tell you when to hold 2x/3x SPY or
2x/3x QQQ", not "does each instrument's own trend tell you when to hold
it" - that per-instrument-signal version is already in FINDINGS.md section A,
over the real-ETF 15y window).
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

D = {t: hist(t) for t in ["^SP500TR", "^NDX", "^IRX"]}
sp, ndx = D["^SP500TR"], D["^NDX"]
irx = D["^IRX"] / 100.0
def rate(idx): return irx.reindex(idx).ffill().fillna(0.02)

SPREAD = 0.005
EXP = {"SSO": .0090, "UPRO": .0091, "QLD": .0095, "TQQQ": .0084}
LEV = {"SSO": 2, "UPRO": 3, "QLD": 2, "TQQQ": 3}
IDX = {"SSO": sp, "UPRO": sp, "QLD": ndx, "TQQQ": ndx}
NDX_DIV = 0.007

def synth_ret(name):
    idx_series = IDX[name]
    r = idx_series.pct_change()
    if IDX[name] is ndx: r = r + NDX_DIV / 252
    L, ter = LEV[name], EXP[name]
    fin = rate(r.index) + SPREAD
    return (L * r - (L - 1) * (fin / 252) - ter / 252).rename(name)

RET = {name: synth_ret(name) for name in ["SSO", "UPRO", "QLD", "TQQQ"]}
cash_ret = (rate(sp.index) / 252).rename("cash")

def asym_sma_gate(prices, period, buy_thresh, sell_thresh):
    sma = prices.rolling(window=period, min_periods=period).mean()
    upper = sma * (1.0 + buy_thresh); lower = sma * (1.0 - sell_thresh)
    p, u, l, s = prices.to_numpy(), upper.to_numpy(), lower.to_numpy(), sma.to_numpy()
    out = np.full(len(prices), np.nan); state = np.nan
    for i in range(len(prices)):
        if np.isnan(s[i]) or np.isnan(p[i]): out[i] = np.nan; state = np.nan; continue
        if np.isnan(state): state = 1.0 if p[i] > s[i] else 0.0
        if p[i] > u[i]: state = 1.0
        elif p[i] < l[i]: state = 0.0
        out[i] = state
    return pd.Series(out, index=prices.index)

spr = sp.pct_change()
SIGNALS = {
    "k>=2": F.vote_of_k([F.sma_gate(sp, 250), F.sma_gate(sp, 100),
                          F.realized_vol_gate(spr, 21, 0.40), F.ar1_gate(spr, 30, 0.0)], 2),
    "k>=2 (+/-5% buf)": F.vote_of_k([F.sma_gate(sp, 250, 0.05), F.sma_gate(sp, 100, 0.05),
                                      F.realized_vol_gate(spr, 21, 0.40), F.ar1_gate(spr, 30, 0.0)], 2),
    "SPY200SMA 4%/3%": asym_sma_gate(sp, 200, 0.04, 0.03),
}

START = "1995-01-01"
WIN = {
    "DOT-COM 00-02": ("2000-03-01", "2002-12-31"),
    "GFC 07-09": ("2007-10-01", "2009-06-30"),
    "COVID 2020": ("2020-02-15", "2020-04-30"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
}

rows = []
for inst in ["SSO", "UPRO", "QLD", "TQQQ"]:
    inst_ret = RET[inst]
    for sig_name, sig in SIGNALS.items():
        pos = sig.shift(1).ffill().fillna(0.0)
        idx = inst_ret.index
        strat = (pos.reindex(idx).fillna(0) * inst_ret
                 + (1 - pos.reindex(idx).fillna(0)) * cash_ret.reindex(idx).fillna(0))
        r = strat.loc[START:].dropna()
        eq = (1 + r).cumprod()
        p = pos.reindex(r.index).fillna(0)
        row = {
            "instrument": inst, "trigger": sig_name,
            "cagr": cagr(eq), "maxdd": max_drawdown(eq), "sortino": sortino(r),
            "flips": n_trades(p), "time_in": p.mean(),
        }
        for wname, (lo, hi) in WIN.items():
            rs = strat.loc[lo:hi].dropna()
            if len(rs) < 5: row[wname] = None; continue
            es = (1 + rs).cumprod()
            row[wname] = (es / es.cummax() - 1).min()
        rows.append(row)

print(f"=== 8-way: {{SSO,UPRO,QLD,TQQQ}} x {{k>=2, SPY200SMA 4%/3%}}, SPY-triggered, cash risk-off, "
      f"{START}..2026-08-24 ===\n")
hdr = f"{'instrument':<10}{'trigger':<18}{'CAGR':>8}{'maxDD':>8}{'Sortino':>9}{'flips':>7}{'time-in':>9}"
print(hdr)
for row in rows:
    print(f"{row['instrument']:<10}{row['trigger']:<18}{row['cagr']*100:7.1f}%{row['maxdd']*100:7.0f}%"
          f"{row['sortino']:9.2f}{row['flips']:7d}{row['time_in']*100:8.0f}%")

print(f"\n=== crash-window max drawdown ===")
hdr2 = f"{'instrument':<10}{'trigger':<18}" + "".join(f"{w:>16}" for w in WIN)
print(hdr2)
for row in rows:
    line = f"{row['instrument']:<10}{row['trigger']:<18}"
    for w in WIN:
        v = row[w]
        line += f"{(v*100):15.0f}%" if v is not None else f"{'n/a':>16}"
    print(line)

# reference buy-holds
print("\n=== buy-hold references ===")
for inst in ["SSO", "UPRO", "QLD", "TQQQ"]:
    r = RET[inst].loc[START:].dropna(); e = (1 + r).cumprod()
    print(f"{inst:<10} buy-hold   CAGR {cagr(e)*100:6.1f}%  maxDD {max_drawdown(e)*100:6.0f}%  Sortino {sortino(r):6.2f}")
