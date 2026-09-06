"""New strategy proposal: month-end QQQ/TLT dual-gate de-lever.

Base (both signals healthy): QLD 50 / TLT 10 / GLDM 10 / KMLM 10 / VXUS 10 / SPMO 10
QQQ < 200sma  -> QLD's slot splits evenly across TLT/GLDM/KMLM/VXUS/SPMO
TLT < 200sma  -> TLT's (possibly-already-bumped) slot splits evenly across
                 GLDM/KMLM/VXUS/SPMO (never back into QLD)
Both broken   -> cascades: QLD's 50 first spreads to the other 5 (incl. TLT),
                 then TLT's resulting weight spreads to GLDM/KMLM/VXUS/SPMO
                 -> GLDM/KMLM/VXUS/SPMO 25% each, QLD/TLT both 0%.

Signals are plain SMA200 zero-crossings (no band, as specified), sampled at
month-end and held for the following month (T+1 executed).

KMLM's inception (2020-12-02) binds the window -- this strategy cannot be
tested through COVID 2020 on real data; only the 2022 bear is available as
a real stress check.
"""
import numpy as np
import pandas as pd

from ai_swing.data import get_price_service
from ai_swing.indicators import functions as F
from ai_swing.backtest.metrics import cagr, max_drawdown, sortino, n_trades

ps = get_price_service()


def series(t):
    return ps.get_close_series(t)


REST = ["TLT", "GLDM", "KMLM", "VXUS", "SPMO"]          # everything but QLD
REST_NO_TLT = ["GLDM", "KMLM", "VXUS", "SPMO"]           # everything but QLD/TLT
BASE = {"QLD": 0.50, "TLT": 0.10, "GLDM": 0.10, "KMLM": 0.10, "VXUS": 0.10, "SPMO": 0.10}

SIGNAL_TICKERS = ["QQQ", "TLT"]
ASSET_TICKERS = ["QLD", "TLT", "GLDM", "KMLM", "VXUS", "SPMO"]
ALL_TICKERS = sorted(set(SIGNAL_TICKERS) | set(ASSET_TICKERS))

common_start = str(series("KMLM").dropna().index[0].date())
px = pd.concat({t: series(t) for t in ALL_TICKERS}, axis=1, sort=True).dropna()
px = px.loc[common_start:]
rets = px.pct_change().fillna(0.0)


def monthly_hold(daily_series):
    monthly = daily_series.resample("ME").last()
    return monthly.reindex(daily_series.index, method="ffill")


qqq_gate = F.sma_gate(px["QQQ"], 200, 0.0)
tlt_gate = F.sma_gate(px["TLT"], 200, 0.0)

qqq_ok = monthly_hold(qqq_gate).shift(1).ffill().fillna(1.0)
tlt_ok = monthly_hold(tlt_gate).shift(1).ffill().fillna(1.0)


def weights_for(qqq_healthy: bool, tlt_healthy: bool) -> dict:
    w = dict(BASE)
    if not qqq_healthy:
        qld_w = w.pop("QLD")
        share = qld_w / len(REST)
        for t in REST:
            w[t] = w.get(t, 0.0) + share
        w["QLD"] = 0.0
    if not tlt_healthy:
        tlt_w = w.pop("TLT", 0.0)
        share = tlt_w / len(REST_NO_TLT)
        for t in REST_NO_TLT:
            w[t] = w.get(t, 0.0) + share
        w["TLT"] = 0.0
    return w


idx = rets.index.intersection(qqq_ok.index).intersection(tlt_ok.index)
qqq_ok, tlt_ok, r = qqq_ok.loc[idx], tlt_ok.loc[idx], rets.loc[idx]

strat_ret = pd.Series(0.0, index=idx)
weight_log = []
for dt in idx:
    w = weights_for(qqq_ok.loc[dt] == 1.0, tlt_ok.loc[dt] == 1.0)
    strat_ret.loc[dt] = sum(w.get(t, 0.0) * r.loc[dt, t] for t in ASSET_TICKERS)
    weight_log.append(w["QLD"])

eq = (1 + strat_ret).cumprod()
pos = qqq_ok * 2 + tlt_ok  # 4-state combo for trade counting

print(f"Window: {px.index[0].date()}..{px.index[-1].date()}  (KMLM-bound, month-end, no bands)\n")
print(f"CAGR {cagr(eq)*100:.1f}%   MaxDD {max_drawdown(eq)*100:.1f}%   "
      f"Sortino {sortino(strat_ret):.2f}   Trades/yr {n_trades(pos)/(len(strat_ret)/252):.1f}")

print("\nTime spent in each state:")
for label, mask in [
    ("Both healthy", (qqq_ok == 1.0) & (tlt_ok == 1.0)),
    ("QQQ broken only", (qqq_ok == 0.0) & (tlt_ok == 1.0)),
    ("TLT broken only", (qqq_ok == 1.0) & (tlt_ok == 0.0)),
    ("Both broken", (qqq_ok == 0.0) & (tlt_ok == 0.0)),
]:
    print(f"  {label:<26} {mask.mean()*100:5.1f}%")

WIN = {"2022 bear": ("2022-01-01", "2022-12-31")}
print("\nCrash-window check (COVID 2020 not testable -- KMLM inception is 2020-12-02):")
for wname, (lo, hi) in WIN.items():
    rs, pos_w = strat_ret.loc[lo:hi], pos.loc[lo:hi]
    es = (1 + rs).cumprod()
    dd = ((es / es.cummax() - 1).min()) * 100
    print(f"  {wname:<12} return {(es.iloc[-1]-1)*100:6.1f}%   maxDD {dd:6.1f}%   flips {n_trades(pos_w)}")

# Benchmark: QLD buy-and-hold, and a naive static 50/10/10/10/10/10 with no gating at all.
qld_bh_eq = (1 + r["QLD"]).cumprod()
static_ret = sum(BASE[t] * r[t] for t in ASSET_TICKERS)
static_eq = (1 + static_ret).cumprod()
print(f"\nBenchmarks, same window:")
print(f"  QLD buy-and-hold        CAGR {cagr(qld_bh_eq)*100:6.1f}%   MaxDD {max_drawdown(qld_bh_eq)*100:6.1f}%")
print(f"  Static (no gate) mix    CAGR {cagr(static_eq)*100:6.1f}%   MaxDD {max_drawdown(static_eq)*100:6.1f}%   "
      f"Sortino {sortino(static_ret):.2f}")
