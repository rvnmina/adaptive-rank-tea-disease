import json, os, time
import torch
from paths import RES
from common import LEVELS, make_model, count_macs
from p3_core import P3Attention, SoftmaxAttention, D
@torch.no_grad()
def median_ms(fn, repeats):
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return sorted(t)[len(t) // 2]
@torch.no_grad()
def bench_all(cases, repeats, rounds=3):
    for fn in cases.values():
        for _ in range(20):
            fn()
    best = {k: float("inf") for k in cases}
    for _ in range(rounds):
        for k, fn in cases.items():
            best[k] = min(best[k], median_ms(fn, repeats))
    return best
def main():
    torch.set_num_threads(1)
    os.makedirs(RES, exist_ok=True)
    x = torch.randn(1, 3, 96, 96)
    models = {"softmax": (make_model("softmax").eval(), None, count_macs("softmax"))}
    for r in LEVELS:
        models[f"p3_r{r}"] = (make_model("p3", r=r).eval(), None, count_macs("p3", r=r))
    for kind in ("nested", "learned"):
        m = make_model(kind).eval()
        for r in LEVELS:
            models[f"{kind}_r{r}"] = (m, r, count_macs(kind, r=r))
    cases = {k: (lambda m=m, r=r: m(x, r=r)) for k, (m, r, _) in models.items()}
    ms = bench_all(cases, repeats=100)
    res = {}
    soft = ms["softmax"]
    print(f"Full model, 96x96 image, 1 CPU thread  (softmax = {soft:.3f} ms)")
    for k, (_, _, mac) in models.items():
        res[k] = {"median_ms": round(ms[k], 3), "macs_M": round(mac["total_M"], 3), "speedup_vs_softmax": round(soft / ms[k], 2)}
        print(f"  {k:<12} {ms[k]:7.3f} ms  {mac['total_M']:5.2f}M MACs  {res[k]['speedup_vs_softmax']:.2f}x")
    print("\nOne attention layer vs token count N  (d=64)")
    print(f"  {'N':>5} {'softmax':>9} {'r=16':>9} {'r=64':>9}  ms")
    layers = {"softmax": SoftmaxAttention(D).eval(), "r16": P3Attention(D, 16).eval(), "r64": P3Attention(D, 64).eval()}
    sweep = {}
    for N in (37, 145, 257, 577, 1025):
        t = torch.randn(1, N, D)
        row = bench_all({k: (lambda a=a: a(t)) for k, a in layers.items()}, repeats=30)
        sweep[N] = {k: round(v, 4) for k, v in row.items()}
        print(f"  {N:>5} {row['softmax']:9.3f} {row['r16']:9.3f} {row['r64']:9.3f}")
    res["token_sweep"] = sweep
    json.dump(res, open(os.path.join(RES, "latency.json"), "w"), indent=2)
    print(f"\nSaved {RES}/latency.json  (this machine's CPU, not the edge device)")
if __name__ == "__main__":
    main()