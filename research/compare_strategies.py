"""Full breakdown: backtest + deploy-score every strategy at a consistent window."""
import json, sys
import httpx

BASE = "http://localhost:8001"
RANGE = 15
c = httpx.Client(base_url=BASE, timeout=180)
c.post("/api/auth/login", json={"email": "admin@example.com", "password": "password"}).raise_for_status()

strats = sorted(c.get("/api/strategies").json(), key=lambda x: x["id"])
out = []
for s in strats:
    sid = s["id"]
    row = {
        "id": sid, "name": s["name"], "bench": s["benchmark_ticker"],
        "risk_on": s.get("risk_on_tickers") or [], "off": s["risk_off_ticker"],
        "k": s["k_threshold"], "inds": [i["name"] for i in s.get("indicators", [])],
        "variants": [], "deploy": None, "tier": None, "error": None,
    }
    try:
        bt = c.post(f"/api/backtest/{sid}?range_years={RANGE}").json()
        row["start"] = bt.get("range_start"); row["end"] = bt.get("range_end")
        row["bench_cagr"] = (bt.get("metrics_benchmark") or {}).get("cagr")
        row["bench_dd"] = (bt.get("metrics_benchmark") or {}).get("max_dd")
        for v in bt.get("variants", []):
            ms = v.get("metrics_strategy") or {}
            mr = v.get("metrics_riskon") or {}
            row["variants"].append({
                "ticker": v.get("risk_on_ticker"),
                "cagr_net": ms.get("cagr_net"), "cagr": ms.get("cagr"),
                "max_dd": ms.get("max_dd"), "sortino_net": ms.get("sortino_net"),
                "sortino": ms.get("sortino"), "n_trades": ms.get("n_trades"),
                "hit": ms.get("hit_rate_vs_benchmark"),
                "bh_cagr": mr.get("cagr"), "bh_dd": mr.get("max_dd"),
            })
        ds = c.get(f"/api/strategies/{sid}/deploy-score?range_years={RANGE}&fresh=true").json()
        row["deploy"] = ds.get("total"); row["tier"] = ds.get("tier_label")
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    out.append(row)
    print(f"done id {sid} {s['name']}", file=sys.stderr)

print(json.dumps(out, indent=1))
