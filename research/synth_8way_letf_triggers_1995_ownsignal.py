"""Corrected version of synth_8way_letf_triggers_1995.py: the k>=2 trigger
now uses each instrument's OWN underlying (SPY for SSO/UPRO, QQQ/NDX for
QLD/TQQQ) instead of SPY for all four. This is what the r/LETFs "40-year
LETF rotation backtest" post's Tier 3 winner does (vote-of-2 on the QQQ
underlying, gating QLD) and what FINDINGS.md section A's real-ETF study
already did -- the earlier sweep's SPY-for-everything k>=2 was a different,
intentional question ("does SPY's trend time a QQQ position"), not a bug,
but it's not the config anyone is actually asking about here.

Risk-off is CASH throughout (no ZROZ/TLT -- decided against after the
reconciliation check showed the residual edge is mostly the 1969-2026
history the post used plus real ZROZ's stronger convexity than any proxy
we can build back that far without more work).

The SPY200SMA(4%/3%) trigger stays SPY-driven for all four instruments on
purpose -- that's the "strat 1" family (use SPY's broad-market trend to
time entries into 2x/3x SPY *or* QQQ positions), a deliberate cross-asset
design already validated earlier in this session, not something to
"correct" to per-instrument.
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
UNDERLYING = {"SSO": sp, "UPRO": sp, "QLD": ndx, "TQQQ": ndx}  # own-underlying signal source
NDX_DIV = 0.007

def synth_ret(name):
    idx_series = UNDERLYING[name]
    r = idx_series.pct_change()
    if idx_series is ndx: r = r + NDX_DIV / 252
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

def k2_signal(underlying, buf):
    r = underlying.pct_change()
    return F.vote_of_k([F.sma_gate(underlying, 250, buf), F.sma_gate(underlying, 100, buf),
                         F.realized_vol_gate(r, 21, 0.40), F.ar1_gate(r, 30, 0.0)], 2)

SPY_200SMA = asym_sma_gate(sp, 200, 0.04, 0.03)

SIGNALS_PER_INST = {}
for inst in ["SSO", "UPRO", "QLD", "TQQQ"]:
    u = UNDERLYING[inst]
    SIGNALS_PER_INST[inst] = {
        "k>=2 (own signal)": k2_signal(u, 0.0),
        "k>=2 (own signal, +/-5% buf)": k2_signal(u, 0.05),
        "SPY200SMA 4%/3%": SPY_200SMA,
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
    for sig_name, sig in SIGNALS_PER_INST[inst].items():
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

print(f"=== {{SSO,UPRO,QLD,TQQQ}} x {{k>=2 own-signal (strict / +-5% buf), SPY200SMA 4%/3%}}, "
      f"cash risk-off, {START}..2026-08-24 ===\n")
hdr = f"{'instrument':<10}{'trigger':<28}{'CAGR':>8}{'maxDD':>8}{'Sortino':>9}{'flips':>7}{'time-in':>9}"
print(hdr)
for row in rows:
    print(f"{row['instrument']:<10}{row['trigger']:<28}{row['cagr']*100:7.1f}%{row['maxdd']*100:7.0f}%"
          f"{row['sortino']:9.2f}{row['flips']:7d}{row['time_in']*100:8.0f}%")

print(f"\n=== crash-window max drawdown ===")
hdr2 = f"{'instrument':<10}{'trigger':<28}" + "".join(f"{w:>16}" for w in WIN)
print(hdr2)
for row in rows:
    line = f"{row['instrument']:<10}{row['trigger']:<28}"
    for w in WIN:
        v = row[w]
        line += f"{(v*100):15.0f}%" if v is not None else f"{'n/a':>16}"
    print(line)

print("\n=== buy-hold references ===")
for inst in ["SSO", "UPRO", "QLD", "TQQQ"]:
    r = RET[inst].loc[START:].dropna(); e = (1 + r).cumprod()
    print(f"{inst:<10} buy-hold   CAGR {cagr(e)*100:6.1f}%  maxDD {max_drawdown(e)*100:6.0f}%  Sortino {sortino(r):6.2f}")
