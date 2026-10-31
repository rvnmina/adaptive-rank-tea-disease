import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data
from paths import RES, FIG
from viz import COLORS, RANK_COLORS, setup_style, chance_line, label_bars
from common import LEVELS, load_crossval, count_macs
INK2 = "#52514e"
def load(name):
    p = os.path.join(RES, name)
    if os.path.isfile(p):
        return json.load(open(p))
    print(f"  [skip] {name} not found")
    return None
def save(name):
    os.makedirs(FIG, exist_ok=True)
    plt.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight")
    plt.close()
    print(f"  saved {name}.png / .pdf")
def cv_series():
    out = {}
    p3 = [(r, load_crossval(f"p3_r{r}")) for r in LEVELS]
    p3 = [(r, c["summary"][str(r)]) for r, c in p3 if c]
    if p3:
        out["p3"] = ([r for r, _ in p3], [s["mean"] for _, s in p3], [s["std"] for _, s in p3])
    for k in ("nested", "learned"):
        c = load_crossval(k)
        if c:
            s = c["summary"]
            out[k] = (LEVELS, [s[str(r)]["mean"] for r in LEVELS], [s[str(r)]["std"] for r in LEVELS])
    soft = load_crossval("softmax")
    return out, (soft["summary"]["full"] if soft else None)
LABEL = {"p3": "P-3, frozen features (one model per rank)",
         "nested": "Nested, frozen features (one model)",
         "learned": "Nested, learned features (ours, one model)"}
def fig_accuracy_vs_rank():
    print("Fig 1: cross-validated accuracy vs rank")
    series, soft = cv_series()
    if not series:
        return
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if soft:
        ax.axhspan(soft["mean"] - soft["std"], soft["mean"] + soft["std"],
                   color=COLORS["softmax"], alpha=0.10, linewidth=0)
        ax.axhline(soft["mean"], color=COLORS["softmax"], linestyle="--", linewidth=1.5,
                   label=f"Softmax attention ceiling ({soft['mean']:.1f}%)")
    for k, (rs, mu, sd) in series.items():
        ax.errorbar(rs, mu, yerr=sd, fmt="o-", color=COLORS[k], linewidth=2, markersize=7,
                    capsize=4, label=LABEL[k])
        ax.annotate(f"{mu[-1]:.1f}", (rs[-1], mu[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    chance_line(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([str(r) for r in LEVELS])
    ax.set_xlim(6.5, 80)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Rank r")
    ax.set_ylabel("Accuracy (%), 5-fold CV")
    ax.set_title("Learning the feature map lifts bounded-rank accuracy at every rank")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)
    _n = len(data.build_cache(quiet=True)["labels"])
    ax.text(0.5, -0.36, f"points: mean of 5 folds ({_n} images, each tested once); bars: std across folds",
            transform=ax.transAxes, ha="center", fontsize=8, color=INK2)
    save("accuracy_vs_rank")
def fig_accuracy_vs_macs():
    print("Fig 2: accuracy vs compute")
    series, soft = cv_series()
    if not series:
        return
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, (rs, mu, sd) in series.items():
        x = [count_macs("p3", r=r)["total_M"] for r in rs]
        ax.plot(x, mu, "o-", color=COLORS[k], linewidth=2, markersize=7, label=LABEL[k])
        if k == "learned":
            for r, xi, yi in zip(rs, x, mu):
                ax.annotate(f"r={r}", (xi, yi), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=8, color=INK2)
    if soft:
        xs = count_macs("softmax")["total_M"]
        ax.plot([xs], [soft["mean"]], "D", color=COLORS["softmax"], markersize=9,
                label=f"Softmax ({soft['mean']:.1f}%)")
    chance_line(ax)
    ax.set_xlabel("MACs per image (millions)")
    ax.set_ylabel("Accuracy (%), 5-fold CV")
    ax.set_title("Accuracy against compute (96x96 image, N=37 tokens)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    save("accuracy_vs_macs")
def fig_mac_comparison():
    print("Fig 3: attention MACs")
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    names = ["softmax"] + [f"r={r}" for r in LEVELS]
    vals = [count_macs("softmax")["attn_total_M"]] + \
           [count_macs("p3", r=r)["attn_total_M"] for r in LEVELS]
    cols = [COLORS["softmax"]] + [RANK_COLORS[r] for r in LEVELS]
    bars = ax.bar(names, vals, color=cols, width=0.6)
    label_bars(ax, bars, vals, fmt="{:.2f}M")
    ax.set_ylabel("Attention MACs per image (millions)")
    ax.set_title("Attention cost at N=37 tokens: bounded rank is cheaper only for r < N/2")
    save("mac_comparison")
def fig_latency():
    print("Fig 4: latency")
    d = load("latency.json")
    if not d:
        return
    setup_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    names = ["softmax"] + [f"p3_r{r}" for r in LEVELS] + [f"learned_r{r}" for r in LEVELS]
    names = [n for n in names if n in d]
    t = [d[n]["median_ms"] for n in names]
    cols = [COLORS["softmax"] if n == "softmax" else
            (COLORS["p3"] if n.startswith("p3") else COLORS["learned"]) for n in names]
    a1.barh(names, t, color=cols)
    for i, v in enumerate(t):
        a1.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    a1.invert_yaxis()
    a1.set_xlabel("Median latency (ms), batch 1, 1 CPU thread")
    a1.set_title("Full model at 96x96 (N=37)")
    sw = d.get("token_sweep", {})
    if sw:
        Ns = sorted(int(n) for n in sw)
        for key, col, lab in (("softmax", COLORS["softmax"], "softmax"),
                              ("r16", RANK_COLORS[16], "bounded r=16"),
                              ("r64", RANK_COLORS[64], "bounded r=64")):
            a2.plot(Ns, [sw[str(n)][key] for n in Ns], "o-", color=col, linewidth=2, label=lab)
        a2.set_xscale("log")
        a2.set_yscale("log")
        a2.set_xticks(Ns)
        a2.set_xticklabels([str(n) for n in Ns])
        a2.set_xlabel("Tokens N")
        a2.set_ylabel("One attention layer (ms)")
        a2.set_title("Bounded rank wins as the image gets more tokens")
        a2.legend(fontsize=9)
    plt.tight_layout()
    save("latency_comparison")
def fig_battery():
    print("Fig 5: battery simulation")
    d = load("simulation.json")
    if not d:
        return
    pols = [p for p in ["fixed_r8", "fixed_r16", "fixed_r32", "fixed_r64", "adaptive", "cascade"]
            if p in d["periodic_30s"]]
    setup_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(pols))
    for j, (sc, col, lab) in enumerate((("periodic_30s", RANK_COLORS[16], "one image every 30 s"),
                                        ("continuous", RANK_COLORS[64], "images back to back"))):
        base = d[sc]["fixed_r64"]["images"]
        v = [d[sc][p]["images"] / base for p in pols]
        b = a1.bar(x + (j - 0.5) * 0.38, v, 0.36, color=col, label=lab)
        label_bars(a1, b, v, fmt="{:.2f}x")
    a1.axhline(1.0, color="#8a8a82", linestyle=":", linewidth=1.2)
    a1.set_xticks(x)
    a1.set_xticklabels([p.replace("fixed_", "") for p in pols])
    a1.set_ylabel("Images per battery charge (x fixed r=64)")
    a1.set_title("Rank only matters when inference dominates the energy")
    a1.legend(fontsize=9)
    acc = [d["periodic_30s"][p]["accuracy"] for p in pols]
    cols = [RANK_COLORS[int(p[7:])] if p.startswith("fixed") else COLORS[p] for p in pols]
    b = a2.bar(x, acc, 0.6, color=cols)
    label_bars(a2, b, acc)
    chance_line(a2)
    a2.set_xticks(x)
    a2.set_xticklabels([p.replace("fixed_", "") for p in pols])
    a2.set_ylabel("Accuracy (%), 5-fold CV")
    a2.set_ylim(0, 105)
    a2.set_title(f"Accuracy of each policy ({d['model']} model)")
    a2.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.text(0.5, -0.03, "power, battery and nJ-per-MAC constants are assumptions (see simulate.py); "
             "accuracy and MAC counts are measured", ha="center", fontsize=8, color=INK2)
    save("battery_simulation")
def fig_cascade():
    print("Fig 6: cascade")
    d = load("cascade.json")
    if not d:
        return
    rows = sorted(d["thresholds"].values(), key=lambda v: v["threshold"])
    ts = [r["threshold"] for r in rows]
    setup_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    a1.plot(ts, [r["accuracy"] for r in rows], "o-", color=COLORS["cascade"], linewidth=2,
            label=f"cascade r={d['r_low']} -> r={d['r_high']}")
    a1.axhline(d["always_low"], color=RANK_COLORS[d["r_low"]], linestyle="--",
               label=f"always r={d['r_low']} ({d['always_low']:.1f}%)")
    a1.axhline(d["always_high"], color=RANK_COLORS[d["r_high"]], linestyle="--",
               label=f"always r={d['r_high']} ({d['always_high']:.1f}%)")
    a1.set_xlabel("Confidence threshold")
    _n = len(data.build_cache(quiet=True)["labels"])
    a1.set_ylabel(f"Accuracy (%), {_n} held-out images")
    a1.set_title(f"Cascade accuracy ({d['model']} model)")
    a1.legend(fontsize=8)
    esc = [r["escalation_rate"] for r in rows]
    mac = [r["macs_vs_high_pct"] for r in rows]
    a2.plot(ts, mac, "s-", color=COLORS["adaptive"], linewidth=2,
            label=f"compute, % of always r={d['r_high']}")
    a2.plot(ts, esc, "o-", color=COLORS["cascade"], linewidth=2, label="images escalated (%)")
    a2.axhline(100, color="#8a8a82", linestyle=":", linewidth=1.2)
    a2.set_xlabel("Confidence threshold")
    a2.set_ylabel("%")
    a2.set_title("Cost of the cascade")
    a2.legend(fontsize=8)
    plt.tight_layout()
    save("cascade_threshold")
def fig_allocation():
    print("Fig 7: per-layer rank allocation")
    d = load("allocation.json")
    if not d:
        return
    al = d["allocations"]
    keys = list(al)
    setup_style()
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.3), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        a = al[k]["mean_allocation"]
        u = al[k]["budget_per_layer"]
        b = ax.bar(range(len(a)), a, color=COLORS["learned"], width=0.6)
        label_bars(ax, b, a, fmt="{:.0f}")
        ax.axhline(u, color=COLORS["p3"], linestyle="--", linewidth=1.5, label=f"uniform r={u}")
        ax.set_xticks(range(len(a)))
        ax.set_xlabel("Layer")
        ax.set_title(f"budget avg r={u}: {al[k]['acc_allocated']:.1f}% "
                     f"vs uniform {al[k]['acc_uniform']:.1f}%", fontsize=10)
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Rank (mean over 5 fold models)")
    fig.suptitle(f"Energy-based rank allocation ({d['model']} model, "
                 f"n={d['n_images']} held-out images)", y=1.03)
    save("rank_allocation")
def vs64(saved_pct):
    word = "less" if saved_pct >= 0 else "more"
    return f"{abs(saved_pct):.0f}% {word} energy than always r=64"
def fig_examples():
    print("Fig 8: worked examples on real images")
    d = load("examples.json")
    if not d:
        return
    imgs = data.build_cache(quiet=True)["images"]
    ex = d["examples"]
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 10.5), gridspec_kw={"hspace": 0.45})
    s = d["summary"]
    lo = str(s["r_low"])
    for ax, e in zip(axes.flat, ex):
        ax.imshow(imgs[e["image_index"]].permute(1, 2, 0).numpy())
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ok = "correct" if e["cascade_correct"] else "WRONG"
        ax.set_title(f"{e['group']} - {e['true_class']}  (clutter {e['clutter']:.3f})\n"
                     f"r={lo} confidence {e['confidence'][lo]:.2f} -> uses r={e['chosen_rank']}, {ok}\n"
                     f"{vs64(e['energy_saved_vs_r64_pct'])}", fontsize=9)
    fig.suptitle(f"Top: 3 cleanest images.  Bottom: 3 most cluttered.  "
                 f"All {s['n_images']}: {s['accepted_at_low_pct']}% answered at r={lo}, "
                 f"{vs64(s['energy_saved_vs_r64_pct'])}", fontsize=11, y=0.99)
    fig.subplots_adjust(top=0.88, bottom=0.02, left=0.02, right=0.98, wspace=0.08)
    save("examples_summary")
def main():
    os.makedirs(FIG, exist_ok=True)
    print(f"Writing figures to {FIG}/")
    for f in (fig_accuracy_vs_rank, fig_accuracy_vs_macs, fig_mac_comparison, fig_latency,
              fig_battery, fig_cascade, fig_allocation, fig_examples):
        f()
    print("Done.")
if __name__ == "__main__":
    main()