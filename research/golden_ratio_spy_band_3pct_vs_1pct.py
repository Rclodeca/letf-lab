"""Compare SPY SMA200 hysteresis band widths (±1% live vs ±3%) for the Golden
Ratio SPY+TIP de-lever gate, holding everything else fixed at the currently-live
settings (TIP SMA200 ±0.15%, both 200-day SMAs).

research/golden_ratio_robustness_grid.py only wiggled the SPY band across
0.5%-1.5% (found 1% dominates within that range). This checks whether a much
wider band (3%) -- which would cut whipsaw trades further but also delay
entries/exits -- helps or hurts, on the full history and on the two real
stress windows available on 100% real ETF data (COVID 2020, 2022 bear) --
same crash-window methodology as golden_ratio_delever.py /
golden_ratio_tip_band_compare.py.
"""
import numpy as np
import pandas as pd

from ai_swing.data import get_price_service
from ai_swing.indicators import functions as F
from ai_swing.backtest.metrics import cagr, max_drawdown, sortino, n_trades

ps = get_price_service()


def series(t):
    return ps.get_close_series(t)


SPY_SMA, TIP_SMA = 200, 200
TIP_BAND = 0.0015  # live value, unchanged
SPY_BANDS = [0.01, 0.03]  # 1% (live), 3%

GR_ON = {"UPRO": 0.50, "VBR": 1 / 6, "GLD": 1 / 6, "TLT": 1 / 6}
GR_OFF = {"VBR": 1 / 3, "GLD": 1 / 3, "TLT": 1 / 3}
assets = sorted(set(GR_ON) | set(GR_OFF))

spy, tip = series("SPY"), series("TIP")
px = pd.concat({t: series(t) for t in assets}, axis=1, sort=True).dropna()
rets = px.pct_change().fillna(0.0)

tip_gate = F.sma_gate(tip, TIP_SMA, TIP_BAND)


def run(spy_band):
    spy_gate = F.sma_gate(spy, SPY_SMA, spy_band)
    gate = ((spy_gate == 1.0) & (tip_gate == 1.0)).astype(float)
    gate[spy_gate.isna() | tip_gate.isna()] = np.nan
    pos = gate.reindex(px.index).shift(1).ffill().fillna(0.0)
    on_ret = (rets[list(GR_ON)] * pd.Series(GR_ON)).sum(axis=1)
    off_ret = (rets[list(GR_OFF)] * pd.Series(GR_OFF)).sum(axis=1)
    strat_ret = (pos * on_ret + (1 - pos) * off_ret).dropna()
    pos = pos.reindex(strat_ret.index)
    eq = (1 + strat_ret).cumprod()
    return strat_ret, pos, eq


print(f"Window: {px.index[0].date()}..{px.index[-1].date()}  "
      f"(TIP SMA{TIP_SMA} ±{TIP_BAND*100:.2f}% fixed; varying SPY SMA{SPY_SMA} band)\n")

results = {}
print(f"{'SPY band':>10}{'CAGR':>9}{'MaxDD':>9}{'Sortino':>10}{'Trades/yr':>11}{'Time risk-on':>14}")
for b in SPY_BANDS:
    strat_ret, pos, eq = run(b)
    years = len(strat_ret) / 252
    results[b] = (strat_ret, pos, eq)
    print(f"{b*100:>9.1f}%{cagr(eq)*100:8.1f}%{max_drawdown(eq)*100:8.1f}%"
          f"{sortino(strat_ret):10.2f}{n_trades(pos)/years:11.1f}{pos.mean()*100:13.0f}%")

WIN = {"COVID 2020": ("2020-02-15", "2020-04-30"), "2022 bear": ("2022-01-01", "2022-12-31")}
print("\nCrash-window check -- did risk-on/off actually flip, and did it help?")
for b in SPY_BANDS:
    strat_ret, pos, eq = results[b]
    print(f"-- SPY band ±{b*100:.1f}% --")
    for wname, (lo, hi) in WIN.items():
        rs, pos_w = strat_ret.loc[lo:hi], pos.loc[lo:hi]
        if len(rs) < 5:
            continue
        es = (1 + rs).cumprod()
        dd = ((es / es.cummax() - 1).min()) * 100
        print(
            f"  {wname:<12} return {(es.iloc[-1]-1)*100:6.1f}%   maxDD {dd:6.1f}%   "
            f"time risk-on {pos_w.mean()*100:4.0f}%   flips {n_trades(pos_w)}"
        )
