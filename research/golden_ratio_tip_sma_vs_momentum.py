"""Compare the live TIP SMA200 +/-0.2% gate against a TIP 3/6/12-month and
1/3/6/12-month momentum gate for the Golden Ratio de-lever signal, holding
the SPY side (SMA200 +/-1%) fixed and varying only the TIP signal.

The momentum score is avg(trailing returns over the given offsets) --
byte-for-byte the same scoring formula `ai_swing.scoring.rotation_3of5.score`
uses for the 3-of-5 rotation strategy -- applied here as a trend filter on
TIP itself (score > 0 = risk-on) instead of the current SMA200 filter.

Tests both a bare zero-crossing (no hysteresis) and a couple of symmetric
bands on the score itself (score > +band => on, score < -band => off,
hold inside) -- same hysteresis idea as F.sma_gate/F._band_gate, just
additive around zero instead of multiplicative around a moving SMA.

Runs on two asset-mix/window configs so a result isn't just an artifact of
one short window:
  - Full 6 (SSO+SPMO+VBR+DBMF+GLD+TLT), DBMF-bound, 2019-05-08+
  - Variant C (UPRO+VBR+GLD+TLT, no SPMO/DBMF), UPRO-bound, 2009+
  (both per golden_ratio_full6.py / golden_ratio_delever.py)
"""
import numpy as np
import pandas as pd

from ai_swing.data import get_price_service
from ai_swing.indicators import functions as F
from ai_swing.backtest.metrics import cagr, max_drawdown, sortino, n_trades

# Frozen at the trading-day-offset convention this analysis was actually run
# with -- ai_swing.scoring.rotation_3of5's own RETURN_OFFSETS has since been
# replaced by calendar-month lookback (MONTHS_BACK), which isn't what this
# script's momentum_score() implements below.
RETURN_OFFSETS = (63, 126, 252)  # ~3m, ~6m, ~12m trading days

ps = get_price_service()


def series(t):
    return ps.get_close_series(t)


def momentum_score(prices, offsets=RETURN_OFFSETS):
    return sum(prices / prices.shift(k) - 1 for k in offsets) / len(offsets)


def score_gate(score, band=0.0):
    """Hysteresis gate on a raw score around zero (additive, not multiplicative
    -- score is already a return, unlike SMA which is a price level)."""
    if band <= 0:
        gate = (score > 0).astype(float)
        gate[score.isna()] = np.nan
        return gate
    out = np.full(len(score), np.nan)
    state = np.nan
    s = score.to_numpy()
    for i in range(len(score)):
        if np.isnan(s[i]):
            out[i] = np.nan
            state = np.nan
            continue
        if np.isnan(state):
            state = 1.0 if s[i] > 0 else 0.0
        if s[i] > band:
            state = 1.0
        elif s[i] < -band:
            state = 0.0
        out[i] = state
    return pd.Series(out, index=score.index)


SPY_SMA, SPY_BAND = 200, 0.01          # live, unchanged
TIP_SMA, TIP_SMA_BAND = 200, 0.002     # live TIP gate (this script's baseline)
MOM_BANDS = [0.0, 0.01, 0.02]          # additive bands on the momentum score

CONFIGS = {
    "Full 6 (SSO+SPMO+VBR+DBMF+GLD+TLT), DBMF-bound": {
        "on": {"SSO": 0.50, "SPMO": 0.10, "VBR": 0.10, "DBMF": 0.10, "GLD": 0.10, "TLT": 0.10},
        "off": {"SPMO": 0.20, "VBR": 0.20, "DBMF": 0.20, "GLD": 0.20, "TLT": 0.20},
        "start_ticker": "DBMF",
    },
    "Variant C (UPRO+VBR+GLD+TLT), UPRO-bound": {
        "on": {"UPRO": 0.50, "VBR": 1 / 6, "GLD": 1 / 6, "TLT": 1 / 6},
        "off": {"VBR": 1 / 3, "GLD": 1 / 3, "TLT": 1 / 3},
        "start_ticker": "UPRO",
    },
}

spy, tip = series("SPY"), series("TIP")
spy_gate = F.sma_gate(spy, SPY_SMA, SPY_BAND)
tip_sma_gate = F.sma_gate(tip, TIP_SMA, TIP_SMA_BAND)
tip_mom_score_3_6_12 = momentum_score(tip)
tip_mom_score_1_3_6_12 = momentum_score(tip, offsets=(21,) + RETURN_OFFSETS)

mom_variants = {"SMA200 +/-0.2% (live)": tip_sma_gate}
for label_prefix, score in [("3/6/12m", tip_mom_score_3_6_12), ("1/3/6/12m", tip_mom_score_1_3_6_12)]:
    for b in MOM_BANDS:
        label = f"Mom({label_prefix}) +/-{b*100:.0f}%" if b > 0 else f"Mom({label_prefix}) zero-cross"
        mom_variants[label] = score_gate(score, b)

WIN = {"COVID 2020": ("2020-02-15", "2020-04-30"), "2022 bear": ("2022-01-01", "2022-12-31")}


def run(px, rets, weights_on, weights_off, tip_gate):
    gate = ((spy_gate == 1.0) & (tip_gate == 1.0)).astype(float)
    gate[spy_gate.isna() | tip_gate.isna()] = np.nan
    pos = gate.reindex(px.index).shift(1).ffill().fillna(0.0)
    on_ret = (rets[list(weights_on)] * pd.Series(weights_on)).sum(axis=1)
    off_ret = (rets[list(weights_off)] * pd.Series(weights_off)).sum(axis=1)
    strat_ret = (pos * on_ret + (1 - pos) * off_ret).dropna()
    pos = pos.reindex(strat_ret.index)
    eq = (1 + strat_ret).cumprod()
    return strat_ret, pos, eq


for cfg_name, cfg in CONFIGS.items():
    assets = sorted(set(cfg["on"]) | set(cfg["off"]))
    common_start = str(series(cfg["start_ticker"]).dropna().index[0].date())
    px = pd.concat({t: series(t) for t in assets}, axis=1, sort=True).dropna()
    px = px.loc[common_start:]
    rets = px.pct_change().fillna(0.0)

    print(f"\n{'='*90}\n{cfg_name}")
    print(f"Window: {px.index[0].date()}..{px.index[-1].date()}  "
          f"(SPY SMA{SPY_SMA} +/-{SPY_BAND*100:.0f}% fixed; varying TIP signal)\n")

    results = {}
    print(f"{'TIP signal':<26}{'CAGR':>9}{'MaxDD':>9}{'Sortino':>10}{'Trades/yr':>11}{'Time risk-on':>14}")
    for name, tip_gate in mom_variants.items():
        strat_ret, pos, eq = run(px, rets, cfg["on"], cfg["off"], tip_gate)
        years = len(strat_ret) / 252
        results[name] = (strat_ret, pos, eq)
        print(f"{name:<26}{cagr(eq)*100:8.1f}%{max_drawdown(eq)*100:8.1f}%"
              f"{sortino(strat_ret):10.2f}{n_trades(pos)/years:11.1f}{pos.mean()*100:13.0f}%")

    print("\nCrash-window check -- did risk-on/off actually flip, and did it help?")
    for name in mom_variants:
        strat_ret, pos, eq = results[name]
        print(f"-- {name} --")
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

    print("\nSignal agreement vs live SMA200 +/-0.2% gate (days where risk state differs):")
    sma_pos = results["SMA200 +/-0.2% (live)"][1]
    for name in mom_variants:
        if name.startswith("SMA200"):
            continue
        pos = results[name][1]
        idx = sma_pos.index.intersection(pos.index)
        disagree = (sma_pos.loc[idx] != pos.loc[idx]).mean() * 100
        print(f"  {name:<26} disagree {disagree:5.1f}% of days")
