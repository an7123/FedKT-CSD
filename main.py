#!/usr/bin/env python3
"""
Minimal DP synthetic image generation via latent sufficient statistics.

Pipeline
--------
  1. Encode private images  ->  latent codes   (pretrained DC-AE, public)
  2. L2-clip to threshold R                    (data-independent)
  3. Per-class mean + cov with Gaussian noise   (eps, delta)-DP
  4. Sample latents from DP Gaussians, decode   (post-processing, free)
  5. Train classifier on synthetic, eval real   (post-processing, free)

Usage
-----
  python main.py --dataset eurosat  --epsilon 10
  python main.py --dataset cifar100 --epsilon 10 --clip-norm 50
"""

import argparse
import json
import math
import os
import zipfile
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from codec import load_dcae
from dp import clip_latents, dp_statistics, sample


# ── Data loading ──────────────────────────────────────────────

# Optional: pre-built EuroSAT zips (images + dataset.json manifest).
# Set via --data-dir or EUROSAT_DATA_DIR env var; otherwise auto-downloaded.
EUROSAT_ZIP_DIR = os.environ.get("EUROSAT_DATA_DIR", "")


def _load_zip(data_dir, split, img_size):
    """Load from preprocessed zip format (optional fast path)."""
    path = os.path.join(data_dir, f"{split}_{img_size}.zip")
    if not os.path.exists(path):
        path = os.path.join(data_dir, f"{split}_32.zip")
    tfm = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    images, labels = [], []
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("dataset.json"))
        for fname, label in meta["labels"]:
            images.append(tfm(Image.open(BytesIO(zf.read(fname))).convert("RGB")))
            labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def _load_eurosat_torchvision(data_dir, img_size, test_frac=4000, seed=0):
    """Auto-download EuroSAT via torchvision and create a deterministic split.

    Uses a fixed split (23000 train / 4000 test) so that
    numbers are directly comparable across runs.
    """
    tfm = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    ds = torchvision.datasets.EuroSAT(root=data_dir, transform=tfm, download=True)
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    test_idx = perm[:test_frac].tolist()
    train_idx = perm[test_frac:].tolist()

    def _gather(idx):
        xs, ys = zip(*(ds[i] for i in idx))
        return torch.stack(xs), torch.tensor(ys)

    tr_x, tr_y = _gather(train_idx)
    te_x, te_y = _gather(test_idx)
    return tr_x, tr_y, te_x, te_y


def load_dataset(name, data_dir="./data", img_size=64):
    """Returns (train_x, train_y, test_x, test_y, num_classes)."""
    if name == "cifar100":
        train = torchvision.datasets.CIFAR100(data_dir, True, download=True)
        test = torchvision.datasets.CIFAR100(data_dir, False, download=True)
        def to_t(ds):
            x = torch.tensor(ds.data).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
            return x, torch.tensor(ds.targets)
        tr_x, tr_y = to_t(train)
        te_x, te_y = to_t(test)
        return tr_x, tr_y, te_x, te_y, 100

    # EuroSAT: prefer preprocessed zips if present (faster and matches the
    # original split exactly), otherwise auto-download via torchvision.
    if data_dir == "./data" and EUROSAT_ZIP_DIR and os.path.exists(
            os.path.join(EUROSAT_ZIP_DIR, f"train_{img_size}.zip")):
        tr_x, tr_y = _load_zip(EUROSAT_ZIP_DIR, "train", img_size)
        te_x, te_y = _load_zip(EUROSAT_ZIP_DIR, "test", img_size)
    elif os.path.exists(os.path.join(data_dir, f"train_{img_size}.zip")):
        tr_x, tr_y = _load_zip(data_dir, "train", img_size)
        te_x, te_y = _load_zip(data_dir, "test", img_size)
    else:
        tr_x, tr_y, te_x, te_y = _load_eurosat_torchvision(data_dir, img_size)
    return tr_x, tr_y, te_x, te_y, tr_y.unique().numel()


# ── Encode / decode ───────────────────────────────────────────

@torch.no_grad()
def encode_all(encoder, images, bs=256, device="cuda"):
    return torch.cat([encoder(images[i:i+bs].to(device)).cpu()
                      for i in range(0, len(images), bs)])


@torch.no_grad()
def decode_all(decoder, z, shape, bs=256, device="cuda"):
    return torch.cat([decoder(z[i:i+bs].view(-1, *shape).to(device)).clamp(0, 1).cpu()
                      for i in range(0, len(z), bs)])


# ── Visualization ─────────────────────────────────────────────

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbVeg", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

def visualize(real_x, real_y, syn_x, syn_y, K, out_dir, dataset,
              n_show=8, dpi=150):
    """Save visualization grids comparing real vs DP-synthetic images."""
    class_names = EUROSAT_CLASSES if dataset == "eurosat" else \
        [str(i) for i in range(K)]

    # ── 1. Real vs Synthetic comparison grid (rows = classes) ──
    fig, axes = plt.subplots(K, 2 * n_show, figsize=(2 * n_show * 1.1, K * 1.2))
    fig.suptitle(f"Real (left {n_show}) vs DP-Synthetic (right {n_show})",
                 fontsize=14, y=1.01)

    for c in range(K):
        real_c = real_x[real_y == c]
        syn_c = syn_x[syn_y == c]
        for j in range(n_show):
            # real
            ax = axes[c, j]
            ax.imshow(real_c[j].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.axis("off")
            if j == 0:
                ax.set_title(class_names[c], fontsize=7, loc="left")

            # synthetic
            ax = axes[c, n_show + j]
            ax.imshow(syn_c[j].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.axis("off")

    # add divider labels
    fig.text(0.25, 1.005, "Real", ha="center", fontsize=12, weight="bold",
             transform=fig.transFigure)
    fig.text(0.75, 1.005, "DP-Synthetic", ha="center", fontsize=12,
             weight="bold", transform=fig.transFigure)

    plt.tight_layout()
    path = os.path.join(out_dir, "real_vs_synthetic.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")

    # ── 2. Synthetic-only overview grid (10x10 or smaller) ─────
    n_rows = min(K, 10)
    n_cols = n_show
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.3, n_rows * 1.4))
    fig.suptitle(f"DP-Synthetic Samples ({dataset.upper()})", fontsize=14)

    for r in range(n_rows):
        syn_c = syn_x[syn_y == r]
        for col in range(n_cols):
            ax = axes[r, col] if n_rows > 1 else axes[col]
            ax.imshow(syn_c[col].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(class_names[r], fontsize=7, rotation=0,
                              labelpad=50, va="center")

    plt.tight_layout()
    path = os.path.join(out_dir, "synthetic_grid.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")



# ── Classifier ────────────────────────────────────────────────

def _make_resnet18(num_classes):
    """CIFAR-style ResNet-18: 3x3 conv1, no maxpool (preserves spatial info)."""
    import torch.nn as nn
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def train_and_eval(syn_x, syn_y, te_x, te_y, K, device, epochs=200):
    """Train CIFAR-style ResNet-18 on synthetic data, evaluate on real test."""
    model = _make_resnet18(K).to(device)
    # normalise to [-1, 1]
    syn_x_norm = syn_x * 2 - 1
    te_x_norm = te_x * 2 - 1
    train_ld = DataLoader(TensorDataset(syn_x_norm, syn_y),
                          batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True)
    test_ld = DataLoader(TensorDataset(te_x_norm, te_y),
                         batch_size=128, shuffle=False,
                         num_workers=4, pin_memory=True)

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=60, gamma=0.2)
    best = 0.0

    for ep in range(epochs):
        model.train()
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            # random horizontal flip
            mask = torch.rand(len(x), 1, 1, 1, device=device) < 0.5
            x = torch.where(mask, x.flip(-1), x)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            correct = sum(
                (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
                for x, y in test_ld)
        acc = 100 * correct / len(te_y)
        best = max(best, acc)
        print(f"    epoch {ep+1:>3}: {acc:.1f}%  (best {best:.1f}%)")

    return best


# ── Main ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar100", "eurosat"], default="eurosat")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--epsilon", type=float, default=10.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--clip-norm", type=float, default=34.0,
                   help="Data-independent L2 clip threshold R")
    p.add_argument("--samples-per-class", type=int, default=0,
                   help="Synthetic images per class (0 = match training set)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--dcae-model", default="dc-ae-f32c32-in-1.0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── 1. Data ───────────────────────────────────────────────
    print(f"[1/5] Loading {args.dataset} ...")
    tr_x, tr_y, te_x, te_y, K = load_dataset(args.dataset, args.data_dir)
    n_per_class = len(tr_y) // K
    spc = args.samples_per_class or n_per_class
    print(f"  train: {tuple(tr_x.shape)}, test: {tuple(te_x.shape)}, "
          f"K={K}, n/class~{n_per_class}")

    # ── 2. Encode ─────────────────────────────────────────────
    print(f"[2/5] Encoding with DC-AE ...")
    encoder, decoder = load_dcae(args.dcae_model, device)
    latents = encode_all(encoder, tr_x, device=device)
    lat_shape = tuple(latents.shape[1:])     # (32, 2, 2)
    flat = latents.view(len(latents), -1)    # (N, 128)
    d = flat.shape[1]
    norms = flat.norm(dim=1)
    print(f"  latent: {lat_shape} -> d={d}")

    # ── 3. Clip + DP statistics ───────────────────────────────
    R = args.clip_norm
    print(f"[3/5] DP statistics (eps={args.epsilon}, delta={args.delta}, R={R}) ...")
    clipped = clip_latents(flat, R)
    pct = (norms > R).float().mean()
    print(f"  clipped {pct:.1%} of latents to R={R}")

    stats = dp_statistics(clipped, tr_y, args.epsilon, args.delta, R)

    # privacy summary
    from dp import _sigma as _dp_sigma
    n_min = min(s["n"] for s in stats.values())
    s_mu = R / n_min
    s_cov = R ** 2 / n_min
    sig_mu = _dp_sigma(s_mu, args.epsilon / 2, args.delta / 2)
    sig_cov = _dp_sigma(s_cov, args.epsilon / 2, args.delta / 2)
    print(f"  classes: {K}, n_min: {n_min}")
    print(f"  sens(mean)={s_mu:.6f}  sigma(mean)={sig_mu:.6f}")
    print(f"  sens(cov) ={s_cov:.4f}  sigma(cov) ={sig_cov:.4f}")

    # ── 4. Generate synthetic images ──────────────────────────
    print(f"[4/5] Generating {spc}x{K} = {spc * K} synthetic images ...")
    syn_z, syn_y = sample(stats, spc)
    syn_x = decode_all(decoder, syn_z, lat_shape, device=device)
    print(f"  synthetic: {tuple(syn_x.shape)}")

    torch.save({"images": syn_x, "labels": syn_y},
               os.path.join(args.output_dir, "synthetic.pt"))

    # ── 4b. Visualize ────────────────────────────────────────
    print(f"  Saving visualizations ...")
    visualize(tr_x, tr_y, syn_x, syn_y, K, args.output_dir, args.dataset)
    # norm histogram uses latent norms (not pixel), so save separately
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(norms.numpy(), bins=60, alpha=0.7, color="steelblue",
            edgecolor="white", linewidth=0.5)
    ax.axvline(R, color="red", linestyle="--", linewidth=1.5, label=f"R={R}")
    ax.set_xlabel("Latent L2 Norm")
    ax.set_ylabel("Count")
    ax.set_title("Latent Norm Distribution (before clipping)")
    ax.legend()
    plt.tight_layout()
    norm_path = os.path.join(args.output_dir, "norm_histogram.png")
    fig.savefig(norm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {norm_path}")

    del encoder, decoder
    torch.cuda.empty_cache()

    # ── 5. Train + evaluate ───────────────────────────────────
    print(f"[5/6] Training ResNet-18 ({args.epochs} epochs) ...")
    acc = train_and_eval(syn_x, syn_y, te_x, te_y, K, device, args.epochs)

    print(f"\n{'=' * 55}")
    print(f"  dataset     : {args.dataset}")
    print(f"  DP          : (eps={args.epsilon}, delta={args.delta})-DP")
    print(f"  clip norm R : {R}  (data-independent)")
    print(f"  latent dim  : {d}")
    print(f"  accuracy    : {acc:.1f}%")
    print(f"{'=' * 55}")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"dataset": args.dataset, "epsilon": args.epsilon,
                   "delta": args.delta, "clip_norm": R, "d": d,
                   "accuracy": acc, "K": K, "n_min": n_min}, f, indent=2)


if __name__ == "__main__":
    main()
