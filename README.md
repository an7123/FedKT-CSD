# Minimal DP Synthetic Image Generation

A ~250-line reference implementation of differentially-private synthetic
image generation via **latent sufficient statistics**: encode private images
with a frozen public autoencoder, release per-class Gaussian moments under
the analytic Gaussian mechanism, then sample and decode.

This is the minimal pipeline behind the EuroSAT result reported in the
paper (≈ **82.7% test accuracy at ε=10, δ=1e-5**), with no diffusion
training, no DP-SGD, and no per-example gradients.

## Pipeline

```
private images
   │  encode (frozen DC-AE, public)        ← no privacy cost
   ▼
latent codes  z ∈ ℝ¹²⁸
   │  L2-clip to threshold R               ← data-independent
   ▼
clipped latents
   │  per-class μ, Σ + Gaussian noise      ← (ε, δ)-DP release
   ▼
DP sufficient statistics
   │  sample z' ~ N(μ̂, Σ̂)                 ← post-processing (free)
   │  decode (frozen DC-AE)                ← post-processing (free)
   ▼
synthetic images
   │  train ResNet-18, eval on real test   ← post-processing (free)
   ▼
downstream accuracy
```

Privacy accounting: ε is split evenly across the per-class first and
second moments and calibrated with the **Gaussian mechanism**. 

## Files

| file                   | role                                                          |
|------------------------|---------------------------------------------------------------|
| `main.py`              | centralized driver (load → encode → DP → decode → eval)       |
| `codec.py`             | DC-AE encoder/decoder wrapper (frozen, public)                |
| `dp.py`                | clipping, DP sufficient statistics, sampling                  |
| `fed.py`               | federated variant: same pipeline, split across clients        |
| `verify_invariance.py` | correctness test: federated stats == centralized stats        |

`fed.py` is the federated counterpart of `main.py`. Each client locally
encodes its shard, computes per-class sums of `z` and `zz^T`, adds its
share of Gaussian noise, and uploads to the server, which aggregates and
samples. Because mean/covariance are linear in the per-example sums and
the Gaussian mechanism is closed under summation, this is mathematically
identical to the centralized release — `verify_invariance.py` runs both
paths on the same data with the same RNG seed and asserts the resulting
per-class (μ, Σ) match bit-for-bit. It exists so the federated split can
be trusted without re-doing the privacy analysis.

```bash
python fed.py              --dataset eurosat --num-clients 20 --alpha 0.5
python verify_invariance.py --dataset eurosat
```

## Setup

Requires Python ≥ 3.10, a CUDA GPU, and ~5 GB disk for the DC-AE weights.

```bash
./setup.sh
```

`setup.sh` does three things:

1. `pip install -r requirements.txt`
2. clones [`mit-han-lab/efficientvit`](https://github.com/mit-han-lab/efficientvit)
   into `./efficientvit/` (it's not on PyPI). Override with
   `EFFICIENTVIT_DIR=/path ./setup.sh` to share a clone across projects.
3. pre-fetches the DC-AE weights from
   `mit-han-lab/dc-ae-f32c32-in-1.0` so the first run works offline.

DC-AE uses Triton kernels and currently only runs on `cuda:0`.

The EuroSAT dataset (~90 MB) is auto-downloaded by `torchvision` into
`./data/` on first run. CIFAR-100 likewise.

## Run

```bash
# end-to-end: EuroSAT, ε=10, 200 epochs (≈ paper number)
./run.sh

# or call main.py directly
python main.py --dataset eurosat  --epsilon 10
python main.py --dataset cifar100 --epsilon 10 --clip-norm 50
```

`run.sh` reads `DATASET`, `EPSILON`, `EPOCHS`, `DATA_DIR`, `OUTPUT_DIR`
from the environment, e.g. `EPSILON=1 EPOCHS=200 ./run.sh`.

Useful flags:

| flag                  | default | meaning                                    |
|-----------------------|---------|--------------------------------------------|
| `--dataset`           | eurosat | `eurosat` or `cifar100`                    |
| `--epsilon`           | 10      | total DP budget                            |
| `--delta`             | 1e-5    | DP failure probability                     |
| `--clip-norm`         | 50      | data-independent L2 clip threshold R       |
| `--samples-per-class` | 0       | synthetic images per class (0 = match real)|
| `--epochs`            | 200     | downstream classifier epochs               |
| `--output-dir`        | ./output| where to write images, plots, results.json |

If you have pre-built EuroSAT zips (images + `dataset.json` manifest), point to them
with `--data-dir /path/to/eurosat` or `export EUROSAT_DATA_DIR=/path/to/eurosat`.
Otherwise, the dataset is auto-downloaded via torchvision on first run.

## Outputs

After running, `output/` contains:

- `synthetic.pt` — `{images, labels}` tensor of decoded DP samples
- `real_vs_synthetic.png` — side-by-side grid per class
- `synthetic_grid.png` — synthetic-only overview
- `norm_histogram.png` — latent L2 norm distribution + clip threshold
- `results.json` — final accuracy + run config

## Reproducing the paper number

```bash
DATASET=eurosat EPSILON=10 EPOCHS=200 ./run.sh
```

Expected: ~82% test accuracy on EuroSAT at (ε=10, δ=1e-5).
A 40-epoch smoke test reaches ~70% in a few minutes on a single GPU.
