"""Step 2: synthetic leveraged ETFs through the 2000 dot-com crash and 2008 GFC,
plus the full 2000-2026 stress. Signals reuse the app's indicator functions."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from ai_swing.indicators import functions as F

def hist(t):
    df = yf.Ticker(t).history(period="max", auto_adjust=True)
    if df is None or df.empty: return pd.Series(dtype=float, name=t)
    s = df["Close"].copy(); s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index().rename(t)

D = {t: hist(t) for t in ["^SP500TR","^NDX","^IRX","GLD","TLT"]}
irx = (D["^IRX"]/100.0)
def rate(idx): return irx.reindex(idx).ffill().fillna(0.02)

SPREAD=0.005  # borrow spread above risk-free (calibration: makes it slightly conservative)
EXP={"SSO":.009,"UPRO":.0091,"QLD":.0095,"TQQQ":.0084}
def synth_ret(index_series, L, ter, ndx_div=False):
    r=index_series.pct_change()
    if ndx_div: r=r+0.007/252
    fin=rate(r.index)+SPREAD
    return (L*r - (L-1)*(fin/252) - ter/252).rename(f"{L}x")

sp=D["^SP500TR"]; ndx=D["^NDX"]
ret={
 "S&P 1x": sp.pct_change(), "SSO 2x": synth_ret(sp,2,EXP["SSO"]), "UPRO 3x": synth_ret(sp,3,EXP["UPRO"]),
 "NDX 1x": ndx.pct_change()+0.007/252, "QLD 2x": synth_ret(ndx,2,EXP["QLD"],True), "TQQQ 3x": synth_ret(ndx,3,EXP["TQQQ"],True),
}
cash=(rate(sp.index)/252)

# --- signals (on the total-return index, matching adjusted-close basis) ---
spr=sp.pct_change()
k2=F.vote_of_k([F.sma_gate(sp,250),F.sma_gate(sp,100),F.realized_vol_gate(spr,21,0.40),F.ar1_gate(spr,30,0.0)],2)
sma_qqq=F.sma_gate(ndx,200,0.035)
def strat(letf, sig):
    pos=sig.shift(1).ffill().fillna(0.0)
    idx=letf.index
    return (pos.reindex(idx).fillna(0)*letf + (1-pos.reindex(idx).fillna(0))*cash.reindex(idx).fillna(0))
ret["k>=2 SPY->SSO"]=strat(ret["SSO 2x"], k2)
ret["200SMA QQQ->TQQQ"]=strat(ret["TQQQ 3x"], sma_qqq)

def stats(r, lo, hi):
    rr=r.loc[lo:hi].dropna()
    if len(rr)<5: return None
    eq=(1+rr).cumprod(); dd=(eq/eq.cummax()-1).min()
    return eq.iloc[-1]-1, dd

WIN={"DOT-COM 2000-2002":("2000-03-01","2002-12-31"),"GFC 2007-2009":("2007-10-01","2009-06-30")}
order=["S&P 1x","SSO 2x","UPRO 3x","NDX 1x","QLD 2x","TQQQ 3x","k>=2 SPY->SSO","200SMA QQQ->TQQQ"]
for wname,(lo,hi) in WIN.items():
    print(f"\n=== {wname} ===")
    print(f"{'instrument':<20}{'return':>10}{'max drawdown':>16}")
    for k in order:
        s=stats(ret[k],lo,hi)
        if s: print(f"{k:<20}{s[0]*100:>9.0f}%{s[1]*100:>15.0f}%")

# full 2000-2026 CAGR + maxDD
print("\n=== FULL 2000-01-01 .. 2026-08-24 (incl. both crashes) ===")
print(f"{'instrument':<20}{'CAGR':>8}{'maxDD':>9}")
for k in order:
    r=ret[k].loc["2000-01-01":].dropna(); eq=(1+r).cumprod(); n=len(r)
    cagr=eq.iloc[-1]**(252/n)-1; dd=(eq/eq.cummax()-1).min()
    print(f"{k:<20}{cagr*100:>7.1f}%{dd*100:>8.0f}%")

# 60/20/20 hold through 2008 (real GLD + TLT-as-ZROZ proxy + synth SSO), quarterly
print("\n=== 60/20/20 SSO/GLD/(TLT~ZROZ) quarterly, 2004-2026 (GLD inception) ===")
sso_eq=(1+ret["SSO 2x"]).cumprod()
px=pd.concat({"SSO":sso_eq,"GLD":D["GLD"],"ZROZ~TLT":D["TLT"]},axis=1).dropna()
px=px[px.index>="2004-11-18"]; rr=px.pct_change().fillna(0)
w={"SSO":.6,"GLD":.2,"ZROZ~TLT":.2}; sl=pd.Series(w); prevq=None; eqv=[]
for i,dt in enumerate(px.index):
    if i>0: sl=sl*(1+rr.iloc[i])
    q=(dt.year,(dt.month-1)//3)
    if q!=prevq: t=sl.sum(); sl=pd.Series({k:t*w[k] for k in w}); prevq=q
    eqv.append(sl.sum())
eq=pd.Series(eqv,index=px.index); n=len(eq)
print(f"CAGR {(eq.iloc[-1]**(252/n)-1)*100:.1f}%   maxDD {((eq/eq.cummax()-1).min())*100:.0f}%   ({px.index[0].date()}..{px.index[-1].date()})")
gfc=stats(eq.pct_change(),"2007-10-01","2009-06-30")
print(f"  within GFC: return {gfc[0]*100:.0f}%  maxDD {gfc[1]*100:.0f}%")
