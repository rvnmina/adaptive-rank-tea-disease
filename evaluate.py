import glob, json, os
import torch
from paths import RUNS, RES
from data import get_loaders
from common import LEVELS, load_model, evaluate
def main():
    os.makedirs(RES, exist_ok=True)
    _, va, te = get_loaders(quiet=True)
    files = sorted(glob.glob(os.path.join(RUNS, "*.pt")))
    if not files:
        raise SystemExit(f"No checkpoints in {RUNS}. Run train.py first.")
    out, sweep = {}, {}
    print(f"{'checkpoint':<16}{'val':>8}{'test':>8}")
    for path in files:
        name = os.path.basename(path)[:-3]
        model, kind, r = load_model(path)
        v, t = evaluate(model, va)["acc"], evaluate(model, te)["acc"]
        out[name] = {"model": kind, "r": r, "val_acc": round(v, 1), "test_acc": round(t, 1)}
        print(f"{name:<16}{v:8.1f}{t:8.1f}")
        if hasattr(model, "r_max"):
            sweep[kind] = {r: round(evaluate(model, te, r=r)["acc"], 1) for r in [4] + LEVELS + [48]}
            print("   test by rank: " + "  ".join(f"r{k}={a:.1f}" for k, a in sweep[kind].items()))
    json.dump(out, open(os.path.join(RES, "baselines.json"), "w"), indent=2)
    json.dump(sweep, open(os.path.join(RES, "rank_sweep.json"), "w"), indent=2)
    print(f"\nSaved {RES}/baselines.json and rank_sweep.json (n={len(te.dataset)} test images)")
if __name__ == "__main__":
    main()