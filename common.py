import json, os
import torch
import torch.nn as nn
from p3_core import P3ViT, SoftmaxViT, N_TOKENS, D, DEPTH, MLP_DIM
from adaptive_rank import NestedViT
from data import NUM_CLASSES
from paths import RUNS, RES
MODELS = ("softmax", "p3", "nested", "learned")
LEVELS = [8, 16, 32, 64]
def make_model(model_name, r=32, device="cpu"):
    if model_name == "softmax":
        m = SoftmaxViT(num_classes=NUM_CLASSES)
    elif model_name == "p3":
        m = P3ViT(num_classes=NUM_CLASSES, r=r)
    elif model_name == "nested":
        m = NestedViT(num_classes=NUM_CLASSES, r_max=64)
    elif model_name == "learned":
        m = NestedViT(num_classes=NUM_CLASSES, r_max=64, learnable=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return m.to(device)
def multi_rank(model_name):
    return model_name in ("nested", "learned")
def run_name(model_name, r=None, seed=42):
    return f"p3_r{r}_s{seed}" if model_name == "p3" else f"{model_name}_s{seed}"
def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = make_model(a["model"], r=a["r"], device=device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, a["model"], a["r"]
def load_crossval(name):
    p = os.path.join(RES, f"crossval_{name}.json")
    return json.load(open(p)) if os.path.isfile(p) else None
def adaptive_name():
    for name in ("learned", "nested"):
        if load_crossval(name):
            return name
    raise SystemExit("Run first: python crossval.py --model learned")
def fold_split(f, n_folds=5):
    import data
    labels = data.build_cache(quiet=True)["labels"]
    folds = data._stratified_folds(labels, n_folds, 42, NUM_CLASSES)
    te, va = folds[f], folds[(f + 1) % n_folds]
    tr = torch.cat([x for j, x in enumerate(folds) if j not in (f, (f + 1) % n_folds)])
    return tr, va, te
def plain_loader(idx, batch_size=64):
    import data
    from torch.utils.data import DataLoader
    c = data.build_cache(quiet=True)
    return DataLoader(data._Tensors(c["images"][idx], c["labels"][idx], train=False), batch_size=batch_size, shuffle=False)
def fold_models(name):
    import glob
    paths = sorted(glob.glob(os.path.join(RUNS, "cv", f"{name}_f*.pt")))
    if not paths:
        raise SystemExit(f"No fold checkpoints for {name}. Run: python crossval.py --model {name}")
    return [(int(p.rsplit("_f", 1)[1][:-3]), load_model(p)[0]) for p in paths]
@torch.no_grad()
def evaluate(model, loader, device="cpu", r=None):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images, r=r) if (r is not None and hasattr(model, "r_max")) else model(images)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return {"acc": 100.0 * correct / total, "loss": total_loss / total}
def count_macs(model_name, r=64, N=N_TOKENS, d=D, depth=DEPTH, mlp_dim=MLP_DIM):
    proj = 4 * N * d * d
    attn = 2 * N * N * d if model_name == "softmax" else 4 * N * r * d
    mlp = 2 * N * d * mlp_dim
    per_layer = proj + attn + mlp
    return {"attn_per_layer": attn, "per_layer": per_layer, "total": depth * per_layer, "total_M": depth * per_layer / 1e6, "attn_total_M": depth * attn / 1e6}
if __name__ == "__main__":
    print(f"MACs per image (N={N_TOKENS} tokens, d={D}, depth={DEPTH})")
    s = count_macs("softmax")
    print(f"  softmax      : {s['total_M']:.2f}M total, {s['attn_total_M']:.2f}M attention")
    for r in LEVELS:
        m = count_macs("p3", r=r)
        print(f"  rank r={r:<3}  : {m['total_M']:.2f}M total, {m['attn_total_M']:.2f}M attention")
    print(f"Bounded rank beats softmax on attention only for r < N/2 = {N_TOKENS/2:.1f}")