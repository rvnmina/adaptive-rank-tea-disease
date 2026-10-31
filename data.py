import os, hashlib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from paths import ROOT, CACHE, PV_ROOT
IMG_SIZE = 96
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
_FALLBACK = ["Aphids", "Mite"]

def discover_classes(root=None):
    root = root or PV_ROOT
    if os.path.isdir(root):
        found = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)) and not d.startswith("."))
        if found:
            return found
    return list(_FALLBACK)

CLASSES = discover_classes()
NUM_CLASSES = len(CLASSES)

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

AUG_TF = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
])
NORM = transforms.Normalize(MEAN, STD)

class _Tensors(Dataset):

    def __init__(self, images_u8, labels, train=False):
        self.x, self.y, self.train = images_u8, labels, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i].float() / 255.0
        if self.train:
            img = AUG_TF(img)
        return NORM(img), self.y[i]

def build_cache(root=None, max_per_class=None, quiet=False):
    root = root or PV_ROOT
    os.makedirs(CACHE, exist_ok=True)
    classes = discover_classes(root)
    tag = hashlib.md5(f"{root}_{max_per_class}_{classes}".encode()).hexdigest()[:8]
    cache_path = os.path.join(CACHE, f"ds_{tag}.pt")

    if os.path.isfile(cache_path):
        data = torch.load(cache_path, weights_only=False)
        if not quiet:
            print(f"Cache hit: {cache_path}  ({data['images'].shape[0]} images)")
        return data

    print(f"Building cache from {root} ...")
    imgs, labels = [], []
    for label_idx, cls in enumerate(classes):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            print(f"  WARNING: missing folder {d}")
            continue
        files = sorted(f for f in os.listdir(d)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if max_per_class:
            files = files[:max_per_class]
        for fname in files:
            im = Image.open(os.path.join(d, fname)).convert("RGB")
            t = EVAL_TF(im)
            imgs.append((t * 255).round().to(torch.uint8))
            labels.append(label_idx)
        print(f"  {cls}: {len(files)} images")

    data = {
        "images": torch.stack(imgs),
        "labels": torch.tensor(labels),
        "classes": classes,
    }
    torch.save(data, cache_path)
    print(f"Saved {cache_path} ({os.path.getsize(cache_path)/1e6:.1f} MB)")
    return data

def _stratified_parts(labels, fracs, seed, n_classes):
    g = torch.Generator().manual_seed(seed)
    parts = [[] for _ in fracs]
    for c in range(n_classes):
        idx = (labels == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        n, cuts, acc = len(idx), [], 0
        for f in fracs[:-1]:
            acc += int(round(f * n))
            cuts.append(acc)
        for p, chunk in zip(parts, torch.tensor_split(idx, cuts)):
            p.append(chunk)
    return [torch.cat(p) for p in parts]

def _stratified_folds(labels, k, seed, n_classes):
    g = torch.Generator().manual_seed(seed)
    folds = [[] for _ in range(k)]
    for c in range(n_classes):
        idx = (labels == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        for j, chunk in enumerate(torch.tensor_split(idx, k)):
            folds[j].append(chunk)
    return [torch.cat(f) for f in folds]

def _loaders_from_idx(data, tr, va, te, batch_size, quiet=False):
    X, Y = data["images"], data["labels"]
    mk = lambda idx, train: DataLoader(
        _Tensors(X[idx], Y[idx], train=train),
        batch_size=batch_size, shuffle=train, drop_last=False)
    if not quiet:
        print(f"Split: {len(tr)} train / {len(va)} val / {len(te)} test   "
              f"(batch {batch_size} -> {max(1, (len(tr)+batch_size-1)//batch_size)} train batches)")
    return mk(tr, True), mk(va, False), mk(te, False)

def get_loaders(batch_size=16, seed=42, train_frac=0.70, val_frac=0.15, quiet=False):
    data = build_cache(quiet=quiet)
    n_classes = len(data["classes"])
    tr, va, te = _stratified_parts(
        data["labels"], [train_frac, val_frac, 1 - train_frac - val_frac], seed, n_classes)
    return _loaders_from_idx(data, tr, va, te, batch_size, quiet)

def get_fold_loaders(fold, n_folds=5, batch_size=16, seed=42, quiet=False):
    data = build_cache(quiet=quiet)
    n_classes = len(data["classes"])
    folds = _stratified_folds(data["labels"], n_folds, seed, n_classes)
    te = folds[fold]
    va = folds[(fold + 1) % n_folds]
    tr = torch.cat([f for j, f in enumerate(folds) if j not in (fold, (fold + 1) % n_folds)])
    return _loaders_from_idx(data, tr, va, te, batch_size, quiet)

if __name__ == "__main__":
    d = build_cache()
    print(f"\nimages {tuple(d['images'].shape)}  labels {tuple(d['labels'].shape)}")
    print(f"classes: {d['classes']}")
    for i, name in enumerate(d["classes"]):
        print(f"  {i} {name}: {(d['labels'] == i).sum().item()}")
    print(f"\nrandom-guess baseline = {100/len(d['classes']):.0f}%")