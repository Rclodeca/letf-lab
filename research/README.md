# research/

Backtest experiments and results for LETF rotation vs static-hold strategies.
See **[FINDINGS.md](./FINDINGS.md)** for the results and conclusions.

## Contents

- `FINDINGS.md` — results, methodology, caveats (read this first).
- `strategies_snapshot.json` — a dump of every strategy definition at time of
  research (a durable record, including hand-made ones).
- `create_strategies.py` — idempotently (re)creates the custom strategies via the
  API (BIL risk-off). Run this to rebuild them on a fresh machine.
- `compare_strategies.py` — backtests + deploy-scores every strategy, prints JSON.
- `backtest_holds.py` — static weighted-hold grid (weights × rebalance frequency)
  plus each ticker's inception date. Runs outside the app's Strategy model.

## Reproduce on a fresh machine

```bash
# 1. clone this fork and get the app running locally (see repo README / SETUP.md)
make install
cp backend/.env.example backend/.env      # then set SQLite + a JWT secret
make migrate && make seed                 # seed provides the 4 standard indicators
make dev                                  # backend on :8001

# 2. use the backend's venv so ai_swing + deps are importable
BE=backend/.venv/bin/python

# 3. recreate the custom strategies (idempotent)
$BE research/create_strategies.py

# 4. run the analyses
$BE research/compare_strategies.py > /tmp/breakdown.json      # needs API on :8001
PRICE_CACHE_DIR=./data/prices $BE research/backtest_holds.py  # standalone (price data only)
```

Notes:
- `compare_strategies.py` and `create_strategies.py` talk to the running API on
  `http://localhost:8001` (default seeded login `admin@example.com` / `password`).
- `backtest_holds.py` only needs the price cache + `ai_swing` importable; it does
  not need the API running.
- Backtests use **real ETF price history**, so results start at ETF inception
  (~2009 for the SSO/GLD/ZROZ hold). There is no synthetic pre-inception build.
