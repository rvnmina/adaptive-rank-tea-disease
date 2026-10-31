import glob, json, os
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from paths import RUNS, RES
from p3_core import P3Attention, SoftmaxAttention, N_TOKENS, D
from adaptive_rank import NestedFeatureMap
from common import LEVELS, fold_models, fold_split, plain_loader
def macs_of(module, x):
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            module(x)
    return fc.get_total_flops() // 2
def check_T1():
    N, d = N_TOKENS, D
    x = torch.randn(1, N, d)
    proj = 4 * N * d * d
    rows, ok = {}, True
    soft = macs_of(SoftmaxAttention(d).eval(), x) - proj
    ok &= soft == 2 * N * N * d
    for r in LEVELS:
        got = macs_of(P3Attention(d, r).eval(), x) - proj
        rows[f"r{r}"] = {"counted": got, "formula_4Nrd": 4 * N * r * d, "cheaper_than_softmax": got < soft}
        ok &= got == 4 * N * r * d
    return {"claim": "T1: attention MACs = 4Nrd (bounded) vs 2N^2 d (softmax)", "N": N, "d": d, "softmax_counted": soft, "softmax_formula": 2 * N * N * d, "ranks": rows, "crossover_rank_N_over_2": N / 2, "holds": bool(ok)}
def check_T2():
    x = torch.randn(2, N_TOKENS, D)
    res = {}
    for learnable in (False, True):
        phi = NestedFeatureMap(D, 64, learnable=learnable)
        full = phi(x, 64)
        res["learned" if learnable else "frozen"] = all(torch.allclose(full[..., :r], phi(x, r), atol=1e-6) for r in LEVELS)
    return {"claim": "T2: nested prefix consistency", **res, "holds": all(res.values())}
@torch.no_grad()
def check_T3():
    if not glob.glob(os.path.join(RUNS, "cv", "softmax_f*.pt")):
        return {"claim": "T3: softmax attention is low rank", "holds": False, "note": "run first: python crossval.py --model softmax"}
    per_layer, top = None, {8: [], 16: []}
    for f, model in fold_models("softmax"):
        per_layer = per_layer or [[] for _ in model.blocks]
        for xb, _ in plain_loader(fold_split(f)[2]):
            x = model.embed(xb)
            for i, blk in enumerate(model.blocks):
                S2 = torch.linalg.svdvals(blk.attn.attention_map(blk.norm1(x))) ** 2
                frac = S2.cumsum(-1) / S2.sum(-1, keepdim=True)
                per_layer[i] += ((frac < 0.9).sum(-1) + 1).tolist()
                for r in top:
                    top[r] += frac[:, r - 1].tolist()
                x = blk(x)
    eff = [float(np.mean(v)) for v in per_layer]
    return {"claim": "T3: softmax attention is low rank (90% energy rank < N/2)", "N": N_TOKENS, "rank_for_90pct_energy_per_layer": [round(e, 2) for e in eff], "energy_in_top8_pct": round(100 * float(np.mean(top[8])), 1), "energy_in_top16_pct": round(100 * float(np.mean(top[16])), 1), "n_images": len(per_layer[0]), "holds": max(eff) < N_TOKENS / 2}
def check_T4():
    out = {"claim": "T4: sandwich loss L = CE(full) + CE(sub) + 0.5 KL(sub||full) converges"}
    ok, seen = True, False
    for name in ("nested", "learned"):
        paths = sorted(glob.glob(os.path.join(RUNS, "cv", f"{name}_f*.pt")))
        if not paths:
            continue
        seen = True
        rows = []
        for p in paths:
            log = torch.load(p, map_location="cpu", weights_only=False)["log"]
            rows.append([round(log[0]["train_loss"], 3), round(log[-1]["train_loss"], 3)])
            ok &= log[-1]["train_loss"] < log[0]["train_loss"]
        out[name] = {"first_and_last_loss_per_fold": rows}
    out["holds"] = bool(ok) and seen
    return out
def main():
    os.makedirs(RES, exist_ok=True)
    results = {}
    for name, fn in (("T1", check_T1), ("T2", check_T2), ("T3", check_T3), ("T4", check_T4)):
        r = fn()
        results[name] = r
        print(f"\n--- {name} --- {'PASS' if r['holds'] else 'FAIL'}")
        for k, v in r.items():
            if k != "holds":
                print(f"  {k}: {v}")
    json.dump(results, open(os.path.join(RES, "theory.json"), "w"), indent=2)
    print(f"\n{sum(r['holds'] for r in results.values())}/4 checks hold. Saved {RES}/theory.json")
if __name__ == "__main__":
    main()