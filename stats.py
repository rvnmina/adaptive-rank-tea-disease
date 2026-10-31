import json, math, os
import numpy as np
from paths import RES
from common import LEVELS, load_crossval
def mcnemar_exact(ca, cb):
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)
def paired_t(fa, fb):
    d = np.asarray(fa, float) - np.asarray(fb, float)
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf") if d.mean() else 0.0, (0.0 if d.mean() else 1.0)
    t = d.mean() / (sd / math.sqrt(len(d)))
    try:
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(abs(t), len(d) - 1)
    except ImportError:
        p = float("nan")
    return t, p
def oof(cv, key):
    o = cv["oof"]
    order = np.argsort(o["index"])
    probs = np.asarray(o["probs"][str(key)])[order]
    labels = np.asarray(o["label"])[order]
    return probs.argmax(1) == labels, cv["summary"][str(key)]["folds"]
def main():
    cvs = {n: load_crossval(n) for n in ["learned", "nested", "softmax"] + [f"p3_r{r}" for r in LEVELS]}
    missing = [n for n, v in cvs.items() if v is None]
    if missing:
        print("Missing cross-validation results:", ", ".join(missing))
    pairs = []
    for r in LEVELS:
        pairs += [(("learned", r), ("nested", r)), (("learned", r), (f"p3_r{r}", r)), (("nested", r), (f"p3_r{r}", r))]
    pairs += [(("softmax", "full"), ("learned", 64)), (("softmax", "full"), ("learned", 32))]
    rows = []
    print(f"{'A':<14}{'B':<14}{'acc A':>7}{'acc B':>7}{'diff':>7}{'b':>4}{'c':>4}{'McNemar p':>11}{'t':>7}{'t-test p':>10}")
    for (na, ka), (nb, kb) in pairs:
        if cvs.get(na) is None or cvs.get(nb) is None:
            continue
        ca, fa = oof(cvs[na], ka)
        cb, fb = oof(cvs[nb], kb)
        b, c, p_mc = mcnemar_exact(ca, cb)
        t, p_t = paired_t(fa, fb)
        la = na if na == "softmax" else f"{na.split('_')[0]} r{ka}"
        lb = f"{nb.split('_')[0]} r{kb}"
        row = {"A": la, "B": lb, "acc_A": round(100 * ca.mean(), 1), "acc_B": round(100 * cb.mean(), 1), "b": b, "c": c, "mcnemar_p": round(p_mc, 4), "paired_t": round(float(t), 2), "paired_t_p": round(float(p_t), 4)}
        rows.append(row)
        star = " *" if p_mc < 0.05 else ""
        print(f"{la:<14}{lb:<14}{row['acc_A']:7.1f}{row['acc_B']:7.1f}{row['acc_A']-row['acc_B']:+7.1f}{b:4d}{c:4d}{p_mc:11.4f}{t:7.2f}{p_t:10.4f}{star}")
    json.dump(rows, open(os.path.join(RES, "stats.json"), "w"), indent=2)
    print(f"\n* = significant at p < 0.05 (exact McNemar on {len(ca)} images, 5 folds for t)")
    print(f"Saved {RES}/stats.json")
if __name__ == "__main__":
    main()