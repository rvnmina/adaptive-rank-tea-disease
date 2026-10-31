import json, os
import numpy as np
import torch
from paths import RES
from common import LEVELS, adaptive_name, fold_models, fold_split, plain_loader
@torch.no_grad()
def layer_energy(model):
    return [float((torch.linalg.svdvals(b.attn.phi.W.float()) ** 2).sum()) for b in model.blocks]
def water_filling(energy, budget, levels=LEVELS):
    score = np.asarray(energy) / np.sum(energy)
    alloc = [min(levels, key=lambda l: abs(l - x)) for x in score * budget]
    order = np.argsort(score)
    while sum(alloc) > budget:
        for i in order:
            lower = [l for l in levels if l < alloc[i]]
            if lower:
                alloc[i] = max(lower)
                break
        else:
            break
    while sum(alloc) < budget:
        for i in order[::-1]:
            higher = [l for l in levels if l > alloc[i] and sum(alloc) - alloc[i] + l <= budget]
            if higher:
                alloc[i] = min(higher)
                break
        else:
            break
    return alloc
@torch.no_grad()
def n_correct(model, loader, ranks):
    return sum((model(x, r=ranks).argmax(1) == y).sum().item() for x, y in loader)
def main():
    os.makedirs(RES, exist_ok=True)
    name = adaptive_name()
    budgets = [16, 32]
    tally = {b: {"alloc": 0, "uniform": 0, "allocations": []} for b in budgets}
    shares, n = [], 0
    for f, model in fold_models(name):
        e = layer_energy(model)
        shares.append(100 * np.asarray(e) / np.sum(e))
        te = plain_loader(fold_split(f)[2])
        n += len(te.dataset)
        for b in budgets:
            L = len(e)
            alloc = water_filling(e, b * L)
            tally[b]["allocations"].append(alloc)
            tally[b]["alloc"] += n_correct(model, te, alloc)
            tally[b]["uniform"] += n_correct(model, te, [b] * L)
    share = np.mean(shares, 0)
    print(f"{name} model: feature-map energy share per layer (mean of folds) {np.round(share, 1).tolist()} %")
    out = {"model": name, "energy_share_pct": share.round(2).tolist(), "n_images": n, "allocations": {}}
    for b in budgets:
        t = tally[b]
        row = {"budget_per_layer": b, "per_fold": t["allocations"], "mean_allocation": np.mean(t["allocations"], 0).round(1).tolist(), "acc_allocated": round(100 * t["alloc"] / n, 1), "acc_uniform": round(100 * t["uniform"] / n, 1)}
        out["allocations"][f"avg_r{b}"] = row
        print(f"  avg r={b}: allocated {row['acc_allocated']}%  vs uniform {row['acc_uniform']}%   (fold allocations {t['allocations']})")
    print(f"  n = {n} held-out images")
    json.dump(out, open(os.path.join(RES, "allocation.json"), "w"), indent=2)
    print(f"Saved {RES}/allocation.json")
if __name__ == "__main__":
    main()