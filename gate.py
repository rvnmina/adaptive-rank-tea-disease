import json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from paths import RES
from common import LEVELS, adaptive_name, fold_models, fold_split, plain_loader, count_macs
class RankGate(nn.Module):
    def __init__(self, input_dim, hidden=8, n_levels=len(LEVELS)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_levels)
    def forward(self, f):
        return self.fc2(F.relu(self.fc1(f)))
@torch.no_grad()
def gate_features(model, images):
    x = model.blocks[0](model.embed(images), LEVELS[0])
    return torch.cat([x[:, 0], images.var(dim=(1, 2, 3)).unsqueeze(1)], 1)
@torch.no_grad()
def cheapest_correct(model, images, labels):
    target = torch.full((len(images),), len(LEVELS) - 1)
    for li in reversed(range(len(LEVELS))):
        target[model(images, r=LEVELS[li]).argmax(1) == labels] = li
    return target
def main():
    torch.manual_seed(0)
    os.makedirs(RES, exist_ok=True)
    name = adaptive_name()
    counts = {r: 0 for r in LEVELS}
    correct = total = 0
    macs = 0.0
    for f, model in fold_models(name):
        tr, _, te = fold_split(f)
        xs, ys = next(iter(plain_loader(tr, batch_size=len(tr))))
        X, Y = gate_features(model, xs), cheapest_correct(model, xs, ys)
        gate = RankGate(X.shape[1])
        opt = torch.optim.Adam(gate.parameters(), lr=1e-2, weight_decay=1e-3)
        for _ in range(200):
            loss = F.cross_entropy(gate(X), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            xt, yt = next(iter(plain_loader(te, batch_size=len(te))))
            choice = gate(gate_features(model, xt)).argmax(1)
            for j in range(len(xt)):
                r = LEVELS[choice[j]]
                counts[r] += 1
                correct += int(model(xt[j:j+1], r=r).argmax(1).item() == yt[j].item())
                macs += count_macs("p3", r=r)["total_M"]
                total += 1
    res = {"model": name, "levels": LEVELS, "gate_params": sum(p.numel() for p in gate.parameters()), "rank_distribution": counts, "gated_accuracy": round(100.0 * correct / total, 1), "avg_rank": round(sum(r * c for r, c in counts.items()) / total, 1), "avg_macs_M": round(macs / total, 3), "macs_vs_r64_pct": round(100 * macs / total / count_macs("p3", r=64)["total_M"], 1), "n_images": total}
    json.dump(res, open(os.path.join(RES, "gate.json"), "w"), indent=2)
    print(f"Rank gate on the {name} model ({res['gate_params']} params), {total} held-out images")
    print(f"  accuracy {res['gated_accuracy']}%   avg rank {res['avg_rank']}   compute {res['macs_vs_r64_pct']}% of always r=64")
    print(f"  ranks chosen: {counts}")
    print(f"Saved {RES}/gate.json")
if __name__ == "__main__":
    main()