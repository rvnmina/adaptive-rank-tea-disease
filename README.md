# Energy-Budgeted Adaptive Rank Control for Bounded-Rank Attention on Agricultural Edge Devices

M.Tech project, under the supervision of **Prof. Amlan Chakrabarti**, Department of Computer Science and Engineering, IIT Kharagpur.

Author: **Ravindra Mina**

This repository contains the code, experiment results, and figures for an M.Tech project that extends bounded-rank attention to adaptive, energy-aware rank selection for tea leaf disease classification on edge devices.

---

## Background and the base paper

This project builds directly on the following paper, which is the base of the code in this repository. In the code it is referred to as **P-3** (`--model p3`, `p3_core.py`):

> **Mallick, M. T., Banerjee, S., Mattar, E. A., Bhattacharya, P., Pal, S., Kumar, A., Turdiev, J., Saha, H. N., & Chakrabarti, A.** (2026). **Quantum-inspired quadratic attention with fourier-domain rank control in transformer architectures.** *Scientific Reports*. https://doi.org/10.1038/s41598-026-60978-w

Paper P-3 introduces bounded-rank quadratic attention: a fixed random Fourier feature map of dimension `r` makes the attention rank at most `r`, and the cost scales linearly with sequence length. Its rank `r` is **fixed** and chosen before training, with a separate model for each rank.

## What this project adds

| Area | From P-3 (reused) | New in this project |
|---|---|---|
| Attention form | Bounded-rank attention `Z = phi(Q)(phi(K)^T V)` | Same form, unchanged |
| Feature map | Frozen random Fourier features | **Learnable feature map (`learned`)**, and a **nested frequency ladder (`nested`)** |
| Rank | Fixed `r`, one trained model per rank | **One model serves every rank** (`r = 8, 16, 32, 64`) by slicing the feature map |
| Rank selection | Manual | Energy-budgeted rank allocation, cascade and gate (`allocation.py`, `cascade.py`, `gate.py`) |
| Evaluation | Accuracy on one dataset | Accuracy vs. MACs, latency, battery simulation, worked examples, 6-class tea dataset |
| Theory | Rank bound and complexity | Checks T1 to T4 in `theory_checks.py` (all pass) |

The main contribution is the **learnable, nested feature map**: it lets a single trained model run at any rank, so the device can trade accuracy for energy at run time.

## Dataset

Tea leaf disease image dataset, **2,100 images, 6 classes**, merged from two sources:

| Class | Images |
|---|---|
| Blight | 504 |
| Healthy_Leaf | 357 |
| Helopeltis | 529 |
| Red_Rust | 590 |
| Aphids | 70 |
| Mite | 50 |

Chance accuracy is 16.7%. The class imbalance ratio is 11.8:1 (Mite 50 vs. Red_Rust 590). The Aphids and Mite images are the ones collected for this project; the other four classes come from Prof. Chakrabarti's field data.

## Model and training setup

- Input 96x96, patch size 16, so N = 37 tokens; embedding dimension d = 64, depth 4, MLP width 128
- Training: 100 epochs per run
- Evaluation: 5-fold stratified cross-validation (`crossval.py`), plus a single split for the detailed analyses (`train.py`, `evaluate.py`)
- Models compared:
  - `p3` (P-3 baseline, fixed random features, one model per rank)
  - `nested` (frozen nested features, one model for all ranks)
  - `learned` (learnable nested features, one model for all ranks)
  - `softmax` (full softmax attention, the ceiling)

## Results

### 5-fold cross-validation accuracy (mean ± std, %)

| Model | r = 8 | r = 16 | r = 32 | r = 64 |
|---|---|---|---|---|
| P-3 (fixed, one model per rank) | 47.5 ± 3.0 | 51.1 ± 1.6 | 51.9 ± 1.6 | 52.2 ± 2.2 |
| Nested (one model, all ranks) | 54.1 ± 1.3 | 54.4 ± 2.3 | 55.0 ± 2.8 | 55.3 ± 2.0 |
| **Learned (one model, all ranks)** | **53.8 ± 1.1** | **55.2 ± 1.6** | 55.0 ± 2.6 | 55.0 ± 2.7 |
| Softmax (full attention, no rank) | 68.8 ± 2.3 | | | |

**Reading the table:**

- The learned and nested models beat P-3 at every rank.
- The largest gain is at the lowest rank: r = 8, 53.8% (learned) vs. 47.5% (P-3), a 6.3-point difference. Low rank is where the energy saving matters most.
- P-3 improves steadily with rank (47.5% to 52.2%). The learned and nested models stay flat at about 54 to 55% across all ranks, which is what a single model serving every rank should look like.
- Full softmax attention is still the most accurate (68.8%). The bounded-rank models trade some accuracy for lower compute; they do not beat full attention on accuracy.

Single-split test-set numbers (`evaluate.py`, `stats.py`) are kept as a cross-check only; the cross-validation table above is the reference.

### Comparison with the P-3 paper

The P-3 paper and this project use **different datasets and class counts**, so their accuracy numbers cannot be compared directly.

| | P-3 paper (Mallick et al., 2026) | This project |
|---|---|---|
| Dataset | Tea pest dataset, 19,500 images | Tea leaf disease dataset, 2,100 images |
| Classes | 5 (chance 20%) | 6 (chance 16.7%) |
| Headline accuracy | Bounded-rank attention 84.5% vs. standard ViT 82.1% | Learned, r = 8: 53.8%; softmax 68.8% |
| Attention time per image | 14.8 ms to 5.6 ms | See `results/latency.json` |
| Peak memory | 820 MB to 430 MB | See `results/latency.json` |

What the comparison does support: at the same rank, the learned model is more accurate than P-3, and the gap is largest at low rank. The fair comparison within this project is P-3 vs. learned on the same 2,100 images at the same rank, as in the table above.

## Repository layout

| File | Purpose |
|---|---|
| `paths.py`, `data.py` | Dataset paths and loading |
| `p3_core.py` | P-3 baseline (`p3`) and the softmax reference |
| `adaptive_rank.py` | Learned and nested feature maps (this project's contribution) |
| `common.py`, `train.py`, `crossval.py` | Training and 5-fold cross-validation |
| `evaluate.py`, `stats.py` | Test evaluation, McNemar and paired t-tests |
| `allocation.py`, `cascade.py`, `gate.py` | Energy-budgeted rank allocation, cascade, and gate |
| `theory_checks.py` | Checks T1 to T4 |
| `bench_latency.py`, `simulate.py`, `examples.py` | Latency benchmark, battery simulation, worked examples |
| `data_analysis.py`, `figures.py`, `viz.py` | Dataset analysis and figures |
| `results/` | All JSON outputs |
| `figures/` | Figures as PNG and PDF |

## How to run

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision matplotlib
bash run_all.sh   # runs every script in order
```

Set the dataset path in `paths.py` before running. On a GPU server, run inside `tmux` so the job survives an SSH disconnect.

## Timeline

| Period | Milestone |
|---|---|
| July 2026 | *: Literature survey and project proposal* |
| August 2026 | *TData collection* |
| September 2026 | *baseline P-3 reproduction* |
| 16 October 2026 | Full package delivered: simulation, worked examples, field-trial protocol, paper draft |
| 28 to 29 October 2026 | Learned-feature model developed; 2-class results |
| 30 to 31 October 2026 | Merged 6-class dataset; full pipeline run on the professor's GPU server |
| 31 October 2026 | Code and results published to this repository |

## Acknowledgements

Thanks to **Prof. Amlan Chakrabarti** for supervision, for the field data, and for the P-3 code this project extends. Thanks to the authors of the P-3 paper for the base method.

## Status and limitations

- The field trial on physical edge hardware has not been run yet; the battery demo is simulated.
- Single-split and cross-validation numbers come from one experimental setup (6-class, 2,100 images).
- Full softmax attention remains more accurate than every bounded-rank model.
