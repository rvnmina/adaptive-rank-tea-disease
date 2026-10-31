import json, os
from collections import Counter
import numpy as np
import torch
from PIL import Image
from paths import RES, PV_ROOT
from p3_core import PATCH, D
import data
PRETTY = {
    "Aphids": "Aphids", "Mite": "Red spider mite",
    "Blight": "Blight", "Healthy_Leaf": "Healthy leaf",
    "Helopeltis": "Helopeltis", "Red_Rust": "Red rust",
}
NOTES = {
    "Aphids": ("medium", "small soft-bodied insects, often clustered on the shoot tip; low contrast against young green leaves"),
    "Mite": ("hard", "damage is diffuse bronzing and stippling rather than a discrete lesion, so the symptom has no sharp boundary"),
    "Blight": ("medium", "brown lesions on leaf margins and tips; can resemble drought stress"),
    "Healthy_Leaf": ("easy", "no visible symptoms; baseline reference class"),
    "Helopeltis": ("medium", "discrete circular feeding marks; distinctive but variable severity"),
    "Red_Rust": ("easy", "bright orange-red pustules on leaf underside; highly distinctive"),
}
def _real_resolutions(root, classes, sample=8):
    sizes, modes = Counter(), Counter()
    for cls in classes:
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for f in files[:sample]:
            try:
                with Image.open(os.path.join(d, f)) as im:
                    sizes[im.size] += 1
                    modes[im.mode] += 1
            except Exception:
                pass
    return sizes, modes
def analyze_dataset_stats():
    d = data.build_cache(quiet=True)
    labels = d["labels"]
    classes = d["classes"]
    total = len(labels)
    class_dist = {}
    for i, cls in enumerate(classes):
        n = int((labels == i).sum())
        class_dist[PRETTY.get(cls, cls)] = {
            "count": n,
            "percentage": round(100.0 * n / total, 1),
        }
    x = d["images"].float() / 255.0
    meas_mean = [round(float(x[:, c].mean()), 4) for c in range(3)]
    meas_std = [round(float(x[:, c].std()), 4) for c in range(3)]
    sizes, modes = _real_resolutions(PV_ROOT, classes)
    src_res = ", ".join(f"{w}x{h}" for (w, h), _ in sizes.most_common(3)) or "unknown"
    img_stats = {
        "source_resolution": src_res,
        "resized_to": f"{data.IMG_SIZE}x{data.IMG_SIZE}",
        "colour_mode": ", ".join(m for m, _ in modes.most_common(2)) or "unknown",
        "channels": 3,
        "measured_mean_rgb": meas_mean,
        "measured_std_rgb": meas_std,
        "normalisation_mean": data.MEAN,
        "normalisation_std": data.STD,
        "dtype": "uint8 in cache, float32 after normalisation",
    }
    tr, va, te = data._stratified_parts(labels, [0.70, 0.15, 0.15], 42, len(classes))
    splits = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        per = {PRETTY.get(classes[i], classes[i]): int((labels[idx] == i).sum()) for i in range(len(classes))}
        splits[name] = {"count": int(len(idx)), "per_class": per}
    n_patches = (data.IMG_SIZE // PATCH) ** 2
    patch_stats = {
        "image_size": data.IMG_SIZE,
        "patch_size": PATCH,
        "patches_per_image": n_patches,
        "with_cls_token": n_patches + 1,
        "embed_dim": D,
        "patch_dim": f"{PATCH} x {PATCH} x 3 = {3 * PATCH * PATCH} pixels, projected to d={D}",
    }
    model_input = {
        "batch_shape": f"[B, 3, {data.IMG_SIZE}, {data.IMG_SIZE}]",
        "after_patching": f"[B, {n_patches + 1}, {D}]",
        "cls_token": "learnable, prepended as token 0",
        "position_embedding": f"learnable, added to all {n_patches + 1} tokens",
    }
    difficulty = {}
    for cls in classes:
        level, reason = NOTES.get(cls, ("unknown", "not characterised"))
        difficulty[PRETTY.get(cls, cls)] = {"difficulty": level, "reason": reason}
    counts = [v["count"] for v in class_dist.values()]
    return {
        "dataset_name": "Tea leaf disease (professor's field data + pest images)",
        "total_images": total,
        "num_classes": len(classes),
        "chance_accuracy_pct": round(100.0 / len(classes), 1),
        "imbalance_ratio": round(max(counts) / min(counts), 2),
        "augmentation": "flips, +/-15 deg rotation, 8% translate, 0.9-1.1 scale, colour jitter",
        "class_distribution": class_dist,
        "image_statistics": img_stats,
        "splits": splits,
        "patch_statistics": patch_stats,
        "model_input": model_input,
        "difficulty_estimate": difficulty,
    }
def print_report(s):
    print("=" * 62)
    print("  TEA LEAF DISEASE DATASET ANALYSIS  (all figures measured)")
    print("=" * 62)
    print(f"\nTotal images:      {s['total_images']}")
    print(f"Classes:           {s['num_classes']}")
    print(f"Chance accuracy:   {s['chance_accuracy_pct']}%   <-- the number to beat")
    print(f"Imbalance ratio:   {s['imbalance_ratio']}:1")
    print("\n--- Class Distribution ---")
    for cls, info in s["class_distribution"].items():
        print(f"  {cls:22s}{info['count']:5d}   ({info['percentage']}%)")
    print("\n--- Image Statistics ---")
    i = s["image_statistics"]
    print(f"  Source resolution:  {i['source_resolution']} ({i['colour_mode']})")
    print(f"  Resized to:         {i['resized_to']}")
    print(f"  Measured mean RGB:  {i['measured_mean_rgb']}")
    print(f"  Measured std RGB:   {i['measured_std_rgb']}")
    print(f"  Normalised with:    {i['normalisation_mean']} / {i['normalisation_std']}")
    print("\n--- Train/Val/Test Split (stratified, seed 42) ---")
    for name, info in s["splits"].items():
        per = "  ".join(f"{k} {v}" for k, v in info["per_class"].items())
        print(f"  {name:6s}{info['count']:5d}   ({per})")
    print(f"  Augmentation on train only: {s['augmentation']}")
    print("\n--- Patch Statistics ---")
    p = s["patch_statistics"]
    print(f"  {p['image_size']}x{p['image_size']} image -> {p['patches_per_image']} patches of {p['patch_size']}x{p['patch_size']}")
    print(f"  + CLS token = {p['with_cls_token']} tokens, each d = {p['embed_dim']}")
    print("\n--- Difficulty Estimate ---")
    for cls, info in s["difficulty_estimate"].items():
        print(f"  {cls:22s}[{info['difficulty']:6s}]  {info['reason']}")
def main():
    os.makedirs(RES, exist_ok=True)
    stats = analyze_dataset_stats()
    print_report(stats)
    out = os.path.join(RES, "dataset_analysis.json")
    json.dump(stats, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")
if __name__ == "__main__":
    main()