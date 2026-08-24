"""Hold backtests: weight variants x rebalance frequency, over the MAX available
window. Also prints each ticker's earliest date to explain the history limit."""
import json
import pandas as pd
from ai_swing.data import get_price_service
from ai_swing.backtest.metrics import compute_metrics

ps = get_price_service()


def series(t):
    return ps.get_close_series(t)


# 1. inception (earliest available date) per ticker -> explains the window cap
inception = {}
for t in ["SPY", "QQQ", "SSO", "UPRO", "QLD", "TQQQ", "GLD", "ZROZ", "SGOV", "BIL"]:
    s = series(t).dropna()
    inception[t] = str(s.index[0].date()) if len(s) else "n/a"

PKEY = {
    "A": lambda dt: dt.year,
    "Q": lambda dt: (dt.year, (dt.month - 1) // 3),
    "M": lambda dt: (dt.year, dt.month),
}


def sim(weights, freq):
    px = pd.concat({t: series(t) for t in weights}, axis=1, sort=True).dropna()
    rets = px.pct_change().fillna(0.0)
    tickers = list(weights)
    sleeves = pd.Series({t: weights[t] for t in tickers}, dtype=float)
    key = PKEY[freq]
    prev = key(px.index[0])
    eqv = []
    for i, dt in enumerate(px.index):
        if i > 0:
            sleeves = sleeves * (1.0 + rets.loc[dt, tickers])
        k = key(dt)
        if k != prev:
            total = sleeves.sum()
            sleeves = pd.Series({t: total * weights[t] for t in tickers})
            prev = k
        eqv.append(sleeves.sum())
    eq = pd.Series(eqv, index=px.index)
    m = compute_metrics(eq, eq.pct_change().fillna(0.0))
    return str(px.index[0].date()), str(px.index[-1].date()), m


WEIGHTS = {
    "60/20/20": {"SSO": 0.60, "GLD": 0.20, "ZROZ": 0.20},
    "50/25/25": {"SSO": 0.50, "GLD": 0.25, "ZROZ": 0.25},
}
FREQ = {"A": "annual", "Q": "quarterly", "M": "monthly"}

out = {"inception": inception, "grid": []}
for wname, w in WEIGHTS.items():
    for fk, fname in FREQ.items():
        start, end, m = sim(w, fk)
        out["grid"].append({
            "weights": wname, "rebal": fname, "start": start, "end": end,
            "cagr": round(m.cagr, 4), "max_dd": round(m.max_dd, 4),
            "sortino": round(m.sortino, 4),
        })
print(json.dumps(out, indent=1))
