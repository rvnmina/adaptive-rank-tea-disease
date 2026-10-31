import json, os
from paths import RES
from common import LEVELS, load_crossval, count_macs
BATTERY_WH = 37.0
NJ_PER_MAC = 1.0
ACTIVE_W = 4.0
IDLE_W = 2.5
RESERVE = 0.05
ADAPTIVE = [(0.60, 32), (0.30, 16), (0.0, 8)]
CASCADE_T = 0.8
def inference_joules(r):
    return count_macs("p3", r=r)["total"] * NJ_PER_MAC * 1e-9
def per_image(e_inf, interval_s):
    t = e_inf / ACTIVE_W
    gap = max(interval_s - t, 0.0)
    return e_inf + IDLE_W * gap, max(interval_s, t)
def simulate(acc, cascade, interval_s):
    usable = BATTERY_WH * 3600 * (1 - RESERVE)
    out = {}
    def row(e_inf, accuracy, avg_rank):
        e, dt = per_image(e_inf, interval_s)
        n = usable / e
        return {"images": n, "hours": n * dt / 3600, "accuracy": accuracy, "avg_rank": avg_rank, "joules_per_image": e, "inference_share_pct": 100 * e_inf / e}
    for r in LEVELS:
        out[f"fixed_r{r}"] = row(inference_joules(r), acc[r], r)
    n_tot = h = acc_sum = rank_sum = e_sum = 0.0
    soc = 1.0
    for floor, r in ADAPTIVE:
        lo = max(floor, RESERVE)
        if soc <= lo:
            continue
        e, dt = per_image(inference_joules(r), interval_s)
        n = (soc - lo) * BATTERY_WH * 3600 / e
        n_tot += n
        h += n * dt / 3600
        acc_sum += n * acc[r]
        rank_sum += n * r
        e_sum += n * inference_joules(r)
        soc = lo
    out["adaptive"] = {"images": n_tot, "hours": h, "accuracy": acc_sum / n_tot, "avg_rank": rank_sum / n_tot, "joules_per_image": usable / n_tot, "inference_share_pct": 100 * e_sum / usable}
    if cascade:
        e_inf = inference_joules(cascade["r_low"]) + cascade["escalation_rate"] / 100 * inference_joules(cascade["r_high"])
        out["cascade"] = row(e_inf, cascade["accuracy"], None)
    return out
def main():
    os.makedirs(RES, exist_ok=True)
    name = "learned" if load_crossval("learned") else "nested"
    cv = load_crossval(name)
    if cv is None:
        raise SystemExit("Run first: python crossval.py --model learned")
    acc = {r: cv["summary"][str(r)]["mean"] for r in LEVELS}
    cascade = None
    cp = os.path.join(RES, "cascade.json")
    if os.path.isfile(cp):
        c = json.load(open(cp))
        row = c["thresholds"][f"t{CASCADE_T}"]
        cascade = {"r_low": c["r_low"], "r_high": c["r_high"], "threshold": CASCADE_T, "accuracy": row["accuracy"], "escalation_rate": row["escalation_rate"]}
    results = {"model": name, "assumed": {"battery_Wh": BATTERY_WH, "nJ_per_MAC": NJ_PER_MAC, "active_W": ACTIVE_W, "idle_W": IDLE_W}, "measured": {"accuracy_pct": acc, "cascade": cascade}, "inference_mJ": {r: round(1000 * inference_joules(r), 3) for r in LEVELS}}
    print(f"Inference energy per image (mJ): {results['inference_mJ']}")
    for label, interval in (("periodic_30s", 30.0), ("continuous", 0.0)):
        res = simulate(acc, cascade, interval)
        results[label] = res
        base = res["fixed_r64"]["images"]
        print(f"\n{label}: {'one image every 30 s' if interval else 'images back to back'}  ({name} model)")
        for k, v in res.items():
            print(f"  {k:<10} {v['images']:13,.0f} images  {v['hours']:6.2f} h  acc {v['accuracy']:5.1f}%  {v['images']/base:5.2f}x images of r=64  inference = {v['inference_share_pct']:.2f}% of energy")
    json.dump(results, open(os.path.join(RES, "simulation.json"), "w"), indent=2)
    print(f"\nSaved {RES}/simulation.json")
if __name__ == "__main__":
    main()