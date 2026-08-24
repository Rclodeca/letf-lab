"""Step 1: build synthetic leveraged ETFs from total-return indices and CALIBRATE
against the real ETFs over their overlap period. Reports tracking error so we
know how much to trust pre-inception reconstruction."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

def hist(t, **kw):
    df = yf.Ticker(t).history(period="max", auto_adjust=True, **kw)
    if df is None or df.empty: return pd.Series(dtype=float, name=t)
    s = df["Close"].copy(); s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index().rename(t)

print("fetching indices + short rate + real ETFs ...")
data = {t: hist(t) for t in ["^SP500TR","^NDX","^IRX","SSO","UPRO","QLD","TQQQ"]}
for t,s in data.items():
    print(f"  {t:<8} {('%s..%s (%d rows)'%(s.index[0].date(), s.index[-1].date(), len(s))) if len(s) else 'EMPTY'}")

# short rate (annualized, decimal); ffill; default 2% if missing
irx = (data["^IRX"]/100.0).reindex(pd.date_range(data["^IRX"].index[0], "2026-12-31")).ffill()
def rate_on(idx): return irx.reindex(idx).ffill().fillna(0.02)

EXPENSE = {"SSO":0.0090,"UPRO":0.0091,"QLD":0.0095,"TQQQ":0.0084}
LEV = {"SSO":2,"UPRO":3,"QLD":2,"TQQQ":3}
IDX = {"SSO":"^SP500TR","UPRO":"^SP500TR","QLD":"^NDX","TQQQ":"^NDX"}
NDX_DIV = 0.007  # NDX is price-return; add ~0.7%/yr dividend so it's total-return-ish

def synth(etf):
    idx = data[IDX[etf]]; r = idx.pct_change()
    if IDX[etf]=="^NDX": r = r + NDX_DIV/252  # crude div add for total return
    L, ter = LEV[etf], EXPENSE[etf]
    fin = rate_on(r.index)
    daily = L*r - (L-1)*(fin/252) - ter/252
    return (1.0+daily.fillna(0)).cumprod().rename("synth_"+etf)

print("\n=== CALIBRATION vs real ETFs (overlap) ===")
for etf in ["SSO","UPRO","QLD","TQQQ"]:
    syn = synth(etf); real = data[etf]
    j = pd.concat([syn, real], axis=1, join="inner").dropna()
    j = j[j.index >= real.dropna().index[0]]
    sr = j.iloc[:,0]/j.iloc[:,0].iloc[0]; rr = j.iloc[:,1]/j.iloc[:,1].iloc[0]
    yrs = len(j)/252
    cagr_s = sr.iloc[-1]**(1/yrs)-1; cagr_r = rr.iloc[-1]**(1/yrs)-1
    corr = sr.pct_change().corr(rr.pct_change())
    term_ratio = sr.iloc[-1]/rr.iloc[-1]
    print(f"{etf}: {j.index[0].date()}..{j.index[-1].date()} ({yrs:.1f}y)  "
          f"CAGR synth {cagr_s*100:5.1f}% vs real {cagr_r*100:5.1f}% (gap {(cagr_s-cagr_r)*100:+.1f}pp)  "
          f"daily-corr {corr:.4f}  terminal synth/real {term_ratio:.3f}")
