"""Idempotently (re)create the custom strategies explored in this research.

Run against a running LETF Lab backend (default http://localhost:8001) that has
already been seeded (`make seed`, which provides the 4 standard indicators).
Safe to re-run: skips indicators/strategies that already exist by name.

    python research/create_strategies.py

All strategies use BIL as the risk-off asset (cash-like T-bills, full history
back to 2007 so backtests aren't truncated).
"""
import sys
import httpx

BASE = "http://localhost:8001"
RISK_OFF = "BIL"
c = httpx.Client(base_url=BASE, timeout=120)
c.post("/api/auth/login", json={"email": "admin@example.com", "password": "password"}).raise_for_status()

# --- indicators -----------------------------------------------------------
inds = {i["name"]: i for i in c.get("/api/indicators").json()}
STD = ["SMA250", "SMA100", "Vol21d<40%", "AR(1)_30d>0"]  # seeded vote-of-2 set
std_ids = [inds[n]["id"] for n in STD]


def ensure_indicator(name, params, desc):
    if name in inds:
        return inds[name]["id"]
    r = c.post("/api/indicators", json={"name": name, "type": "SMA_GATE",
                                        "params": params, "description": desc})
    r.raise_for_status()
    iid = r.json()["id"]
    inds[name] = {"id": iid}
    print(f"created indicator {name} -> {iid}", file=sys.stderr)
    return iid


buf35 = ensure_indicator("SMA200 buffer +/-3.5%", {"period": 200, "threshold": 0.035},
                         "200SMA symmetric +/-3.5% band (proxy for +4/-3 traffic light)")
buf40 = ensure_indicator("SMA200 - 4% - Buy", {"period": 200, "threshold": 0.04},
                         "200SMA symmetric +/-4% band")

# --- strategies (name, benchmark, risk_on_ticker, indicator_ids, k) -------
SPECS = [
    ("SPY 2x k>=2",   "SPY", "SSO",  std_ids, 2),
    ("SPY 3x k>=2",   "SPY", "UPRO", std_ids, 2),
    ("QQQ 2x k>=2",   "QQQ", "QLD",  std_ids, 2),
    ("QQQ 3x k>=2",   "QQQ", "TQQQ", std_ids, 2),
    ("SPY 2x 200SMA", "SPY", "SSO",  [buf35], 1),
    ("SPY 3x 200SMA", "SPY", "UPRO", [buf35], 1),
    ("QQQ 2x 200SMA", "QQQ", "QLD",  [buf35], 1),
    ("QQQ 3x 200SMA", "QQQ", "TQQQ", [buf35], 1),
    # id 11 style: use SPY's trend (±4% 200SMA band) to time a 3x QQQ position
    ("200SMA",        "SPY", "TQQQ", [buf40], 1),
]

existing = {s["name"]: s["id"] for s in c.get("/api/strategies").json()}
for name, bench, ticker, ind_ids, k in SPECS:
    if name in existing:
        print(f"exists: {name} (id {existing[name]})", file=sys.stderr)
        continue
    r = c.post("/api/strategies", json={
        "name": name, "benchmark_ticker": bench, "risk_on_tickers": [ticker],
        "risk_off_ticker": RISK_OFF, "k_threshold": k, "indicator_ids": ind_ids,
    })
    r.raise_for_status()
    print(f"created strategy {name} -> id {r.json()['id']}", file=sys.stderr)

print("done.", file=sys.stderr)
