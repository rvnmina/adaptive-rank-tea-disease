import json, os, time

import numpy as np
import torch
import torch.nn.functional as F

import data
from common import LEVELS, make_model, multi_rank, run_name
from paths import RES, RUNS
from train import parser, train_one


@torch.no_grad()
def predict(m, loader, r=None):
    m.eval()
    out = [F.softmax(m(x, r=r) if r is not None else m(x), 1) for x, _ in loader]
    return torch.cat(out)


def main():
    ap = parser()
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(RES, exist_ok=True)
    os.makedirs(os.path.join(RUNS, "cv"), exist_ok=True)
    if a.folds < 3:
        raise SystemExit("--folds must be 3 or more (one fold is test, the next is val)")

    cache = data.build_cache(quiet=True)
    labels, n_cls = cache["labels"], len(cache["classes"])
    folds = data._stratified_folds(labels, a.folds, 42, n_cls)
    keys = LEVELS if multi_rank(a.model) else ([a.r] if a.model == "p3" else ["full"])
    chance = 100.0 / n_cls
    name = run_name(a.model, a.r, a.seed).replace(f"_s{a.seed}", "")
    n_par = sum(p.numel() for p in make_model(a.model, r=a.r).parameters())

    print(f"{a.folds}-fold CV  model={name}  {n_par:,} params  {a.epochs} epochs/fold")
    print(f"chance = {chance:.0f}%  ({n_cls} classes, {len(labels)} images)\n")

    t0 = time.time()
    per_fold, oof_idx, oof_fold = [], [], []
    oof_probs = {str(k): [] for k in keys}
    for f in range(a.folds):
        torch.manual_seed(a.seed + f)
        tr, va, te = data.get_fold_loaders(f, n_folds=a.folds, batch_size=a.bs,
                                           seed=42, quiet=True)
        m, _, log = train_one(a, tr, va, a.seed + f)
        torch.save({"state": m.state_dict(), "args": vars(a), "log": log, "fold": f},
                   os.path.join(RUNS, "cv", f"{name}_f{f}.pt"))
        res = {}
        for k in keys:
            p = predict(m, te, r=k if multi_rank(a.model) else None)
            y = torch.cat([yb for _, yb in te])
            res[str(k)] = 100.0 * (p.argmax(1) == y).float().mean().item()
            oof_probs[str(k)] += [[round(v, 4) for v in row] for row in p.tolist()]
        oof_idx += folds[f].tolist()
        oof_fold += [f] * len(folds[f])
        per_fold.append(res)
        print(f"  fold {f}: " + "  ".join(f"r{k}={v:5.1f}" for k, v in res.items()), flush=True)

    out = {"model": a.model, "name": name, "params": n_par, "folds": a.folds,
           "epochs": a.epochs, "n_classes": n_cls, "chance_pct": chance,
           "per_fold": per_fold, "summary": {},
           "oof": {"index": oof_idx, "fold": oof_fold,
                   "label": labels[oof_idx].tolist(), "probs": oof_probs}}
    print(f"\n{'rank':>6} {'mean':>7} {'std':>7}   vs chance")
    for k in per_fold[0]:
        v = np.array([p[k] for p in per_fold], dtype=float)
        out["summary"][k] = {"mean": round(float(v.mean()), 2), "std": round(float(v.std()), 2),
                             "folds": [round(float(x), 1) for x in v]}
        flag = "ok" if v.mean() - v.std() > chance else "NOT above chance"
        print(f"  r={k:>4} {v.mean():7.1f} {v.std():7.1f}   {flag}")

    path = os.path.join(RES, f"crossval_{name}.json")
    json.dump(out, open(path, "w"))
    print(f"\n[{time.time()-t0:.0f}s]  saved {path}")
    print(f"Every one of the {len(set(oof_idx))} images was in a test fold exactly once.")


if __name__ == "__main__":
    main()