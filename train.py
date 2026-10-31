import argparse, copy, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import data
from common import MODELS, LEVELS, make_model, multi_rank, run_name, evaluate
from paths import RUNS
def sandwich_loss(m, xb, yb, gen, smooth):
    out = m(xb)
    loss = F.cross_entropy(out, yb, label_smoothing=smooth)
    r = LEVELS[int(torch.randint(0, len(LEVELS), (1,), generator=gen))]
    sub = m(xb, r=r)
    loss = loss + F.cross_entropy(sub, yb, label_smoothing=smooth) \
                + 0.5 * F.kl_div(F.log_softmax(sub, 1), F.softmax(out.detach(), 1),
                                 reduction="batchmean")
    return loss
def train_one(a, tr, va, seed, log_every=0):
    m = make_model(a.model, r=a.r)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr, total_steps=a.epochs * len(tr), pct_start=0.25)
    gen = torch.Generator().manual_seed(seed + 7)
    best, best_state, log = -1.0, None, []
    for ep in range(1, a.epochs + 1):
        m.train()
        tot_loss = seen = 0
        for xb, yb in tr:
            if multi_rank(a.model):
                loss = sandwich_loss(m, xb, yb, gen, a.smooth)
            else:
                loss = F.cross_entropy(m(xb), yb, label_smoothing=a.smooth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(yb)
            seen += len(yb)
        rec = {"epoch": ep, "train_loss": tot_loss / seen}
        if multi_rank(a.model):
            rec["val_by_rank"] = {r: evaluate(m, va, r=r)["acc"] for r in LEVELS}
            rec["score"] = sum(rec["val_by_rank"].values()) / len(LEVELS)
        else:
            rec["score"] = evaluate(m, va)["acc"]
        log.append(rec)
        if log_every and (ep % log_every == 0 or ep == a.epochs):
            print(f"  ep {ep:3d}  loss {rec['train_loss']:.3f}  val score {rec['score']:5.1f}")
        if rec["score"] > best:
            best, best_state = rec["score"], copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    return m, best, log
def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="learned", choices=MODELS)
    ap.add_argument("--r", type=int, default=64, help="rank for p3; r_max for nested/learned")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    return ap
def main():
    a = parser().parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    os.makedirs(RUNS, exist_ok=True)

    name = run_name(a.model, a.r, a.seed)
    tr, va, te = data.get_loaders(batch_size=a.bs, seed=42)
    n_par = sum(p.numel() for p in make_model(a.model, r=a.r).parameters())
    print(f"{name}: {n_par:,} params, {len(tr.dataset)} training images")
    t0 = time.time()
    m, best, log = train_one(a, tr, va, a.seed, log_every=10)
    torch.save({"state": m.state_dict(), "args": vars(a), "log": log, "params": n_par},
               os.path.join(RUNS, f"{name}.pt"))
    print(f"  best val score {best:.1f}   [{time.time()-t0:.0f}s]")
    if multi_rank(a.model):
        for r in LEVELS:
            print(f"    test r={r:2d}: {evaluate(m, te, r=r)['acc']:.1f}%")
    else:
        print(f"    test: {evaluate(m, te)['acc']:.1f}%")
    print(f"  saved {os.path.join(RUNS, name)}.pt")
if __name__ == "__main__":
    main()