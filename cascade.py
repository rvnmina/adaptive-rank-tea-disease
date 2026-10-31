import argparse, json, os
import numpy as np
from paths import RES
from common import load_crossval, count_macs
def run(cv, r_low, r_high, thresholds):
    o = cv["oof"]
    y = np.asarray(o["label"])
    p_low, p_high = np.asarray(o["probs"][str(r_low)]), np.asarray(o["probs"][str(r_high)])
    mac_low, mac_high = count_macs("p3", r=r_low)["total_M"], count_macs("p3", r=r_high)["total_M"]
    out = {}
    for t in thresholds:
        esc = p_low.max(1) < t
        pred = np.where(esc, p_high.argmax(1), p_low.argmax(1))
        macs = mac_low + esc.mean() * mac_high
        out[f"t{t}"] = {"threshold": t, "accuracy": round(100 * (pred == y).mean(), 1), "escalation_rate": round(100 * esc.mean(), 1), "avg_macs_M": round(macs, 3), "macs_vs_high_pct": round(100 * macs / mac_high, 1)}
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=["learned", "nested"])
    ap.add_argument("--r_low", type=int, default=8)
    ap.add_argument("--r_high", type=int, default=16)
    a = ap.parse_args()
    name = a.model or ("learned" if load_crossval("learned") else "nested")
    cv = load_crossval(name)
    if cv is None:
        raise SystemExit(f"Run first: python crossval.py --model {name}")
    s = cv["summary"]
    print(f"Cascade on {name} model, r={a.r_low} -> r={a.r_high}, {len(cv['oof']['label'])} held-out images")
    print(f"  always r={a.r_low}: {s[str(a.r_low)]['mean']:.1f}%   always r={a.r_high}: {s[str(a.r_high)]['mean']:.1f}%")
    res = run(cv, a.r_low, a.r_high, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    for v in res.values():
        print(f"  t={v['threshold']:<5} acc={v['accuracy']:5.1f}%  escalated={v['escalation_rate']:5.1f}%  MACs={v['avg_macs_M']:.2f}M ({v['macs_vs_high_pct']:.0f}% of r={a.r_high})")
    out = {"model": name, "r_low": a.r_low, "r_high": a.r_high, "always_low": s[str(a.r_low)]["mean"], "always_high": s[str(a.r_high)]["mean"], "thresholds": res}
    json.dump(out, open(os.path.join(RES, "cascade.json"), "w"), indent=2)
    print(f"Saved {RES}/cascade.json")
if __name__ == "__main__":
    main()