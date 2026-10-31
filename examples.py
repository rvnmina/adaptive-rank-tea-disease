import json, os
import numpy as np
import torch
import torch.nn.functional as F
import data
from paths import RES
from common import LEVELS, load_crossval, count_macs
from simulate import CASCADE_T, inference_joules
def clutter_scores(images_u8):
    g = images_u8.float().mean(1, keepdim=True) / 255.0
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    gx, gy = F.conv2d(g, kx, padding=1), F.conv2d(g, kx.transpose(2, 3), padding=1)
    return (gx ** 2 + gy ** 2).sqrt().mean((1, 2, 3)).numpy()
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
def main():
    os.makedirs(RES, exist_ok=True)
    name = "learned" if load_crossval("learned") else "nested"
    cv = load_crossval(name)
    if cv is None:
        raise SystemExit("Run first: python crossval.py --model learned")
    cache = data.build_cache(quiet=True)
    classes = cache["classes"]
    clutter = clutter_scores(cache["images"])
    o = cv["oof"]
    order = np.argsort(o["index"])
    labels = np.asarray(o["label"])[order]
    probs = {r: np.asarray(o["probs"][str(r)])[order] for r in LEVELS}
    conf = {r: probs[r].max(1) for r in LEVELS}
    joule = {r: inference_joules(r) for r in LEVELS}
    macs = {r: count_macs("p3", r=r)["total_M"] for r in LEVELS}
    cp = os.path.join(RES, "cascade.json")
    c = json.load(open(cp)) if os.path.isfile(cp) else {"r_low": 8, "r_high": 16}
    r_lo, r_hi = c["r_low"], c["r_high"]
    accept = conf[r_lo] >= CASCADE_T
    chosen = np.where(accept, r_lo, r_hi)
    pred = np.where(accept, probs[r_lo].argmax(1), probs[r_hi].argmax(1))
    cost = np.where(accept, joule[r_lo], joule[r_lo] + joule[r_hi])
    picks = list(np.argsort(clutter)[:3]) + list(np.argsort(clutter)[-3:])
    examples = []
    for k, i in enumerate(picks):
        ex = {"image_index": int(i), "group": "clean" if k < 3 else "cluttered", "true_class": classes[labels[i]], "clutter": round(float(clutter[i]), 4), "confidence": {r: round(float(conf[r][i]), 3) for r in LEVELS}, "correct_at": {r: bool(probs[r][i].argmax() == labels[i]) for r in LEVELS}, "chosen_rank": int(chosen[i]), "cascade_correct": bool(pred[i] == labels[i]), "macs_M": round(macs[r_lo] + (0 if accept[i] else macs[r_hi]), 3), "energy_mJ": round(1000 * float(cost[i]), 3), "energy_saved_vs_r64_pct": round(100 * (1 - cost[i] / joule[64]), 1)}
        examples.append(ex)
        print(f"\n[{ex['group']}] image {i}  {ex['true_class']}  clutter={ex['clutter']:.3f}")
        print("  confidence by rank: " + "  ".join(f"r{r}={ex['confidence'][r]:.2f}{'' if ex['correct_at'][r] else '(wrong)'}" for r in LEVELS))
        print(f"  cascade picks r={ex['chosen_rank']}  -> {'correct' if ex['cascade_correct'] else 'WRONG'}   {ex['energy_mJ']:.2f} mJ vs {1000*joule[64]:.2f} mJ for always r=64")
    busy = clutter >= np.median(clutter)
    summary = {"model": name, "threshold": CASCADE_T, "r_low": r_lo, "r_high": r_hi, "n_images": int(len(labels)), "accepted_at_low_pct": round(100 * accept.mean(), 1), "accepted_at_low_clean_half_pct": round(100 * accept[~busy].mean(), 1), "accepted_at_low_cluttered_half_pct": round(100 * accept[busy].mean(), 1), "cascade_accuracy_pct": round(100 * (pred == labels).mean(), 1), "energy_saved_vs_r64_pct": round(100 * (1 - cost.mean() / joule[64]), 1), "spearman_clutter_vs_low_rank_confidence": round(spearman(clutter, conf[r_lo]), 3)}
    print(f"\nAll {summary['n_images']} images, cascade r={r_lo} -> r={r_hi}, threshold {CASCADE_T}:")
    print(f"  answered at r={r_lo}: {summary['accepted_at_low_pct']}%  (clean half {summary['accepted_at_low_clean_half_pct']}%, cluttered half {summary['accepted_at_low_cluttered_half_pct']}%)")
    print(f"  cascade accuracy {summary['cascade_accuracy_pct']}%, energy vs always r=64: {-summary['energy_saved_vs_r64_pct']:+.1f}%")
    print(f"  Spearman(clutter, r={r_lo} confidence) = {summary['spearman_clutter_vs_low_rank_confidence']}")
    json.dump({"examples": examples, "summary": summary}, open(os.path.join(RES, "examples.json"), "w"), indent=2)
    print(f"Saved {RES}/examples.json")
if __name__ == "__main__":
    main()