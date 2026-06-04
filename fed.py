#!/usr/bin/env python3
"""
Federated DP synthetic image generation with simulated secure aggregation.

Each client encodes its private images and sends only per-class sufficient
statistics (sum of clipped latents + sum of outer products + count).
The server securely aggregates the sums, injects DP noise *once*, and
generates a synthetic dataset from the protected class-conditional Gaussians.

Key property
------------
Clients compute SUMS (not means).  Sums are additive:

    S_c = sum_i  sum_{j in client_i, class c}  z_ij
        = sum over all class-c records in the entire dataset

This is identical to the centralized computation.  Therefore the DP
guarantee, noise level, and generation quality are *independent* of the
number of clients and the degree of data heterogeneity (alpha).

SecAgg ensures the server only sees the aggregated total, never any
individual client's contribution.

Privacy model
-------------
Add/remove adjacency: adjacent D ~ D' differ by adding or removing one
record from the dataset.  Sensitivity of the sum query:

    Delta_z  = R     (||z_new||_2 <= R)
    Delta_zz = R^2   (||z_new z_new^T||_F = ||z_new||_2^2 <= R^2)

After dividing by n this gives sensitivity R/n for the mean and R^2/n
for the second moment -- identical to the centralized case in main.py.

DP noise is calibrated via the analytic Gaussian mechanism
(Balle & Wang, NeurIPS 2018) with budget split eps/2 per query.

Usage
-----
  python fed.py --dataset eurosat  --epsilon 10 --num-clients 10
  python fed.py --dataset eurosat  --epsilon 10 --num-clients 50 --alpha 0.1
  python fed.py --dataset cifar100 --epsilon 10 --num-clients 20 --alpha 0.5
"""

import argparse
import json
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
from dp import _sigma, _make_psd, clip_latents, sample


# ── Constants ─────────────────────────────────────────────────

EUROSAT_DIR = os.environ.get("EUROSAT_DATA_DIR", "")
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbVeg", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


# ── Data loading ──────────────────────────────────────────────

def _load_zip(data_dir, split, img_size):
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
    """Auto-download EuroSAT via torchvision and create a deterministic split."""
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

    return _gather(train_idx) + _gather(test_idx)


def load_dataset(name, data_dir="./data", img_size=64):
    """Returns (train_x, train_y, test_x, test_y, num_classes)."""
    if name == "cifar100":
        train = torchvision.datasets.CIFAR100(data_dir, True,  download=True)
        test  = torchvision.datasets.CIFAR100(data_dir, False, download=True)
        def to_t(ds):
            x = torch.tensor(ds.data).permute(0, 3, 1, 2).float() / 255
            return (F.interpolate(x, 64, mode="bilinear", align_corners=False),
                    torch.tensor(ds.targets))
        tr_x, tr_y = to_t(train)
        te_x, te_y = to_t(test)
        return tr_x, tr_y, te_x, te_y, 100

    # Try zip format first (DPImageBench preprocessed), fallback to torchvision
    if data_dir != "./data":
        dd = data_dir
    elif EUROSAT_DIR and os.path.exists(os.path.join(EUROSAT_DIR, "train_64.zip")):
        dd = EUROSAT_DIR
    else:
        dd = data_dir

    zip_path = os.path.join(dd, "train_64.zip")
    if os.path.exists(zip_path):
        tr_x, tr_y = _load_zip(dd, "train", 64)
        te_x, te_y = _load_zip(dd, "test",  64)
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


# ── Federated data partition ──────────────────────────────────

def partition_dirichlet(labels, n_clients, alpha, seed=42):
    """Non-IID Dirichlet partition.

    alpha → 0 : extreme heterogeneity (each client ~ one class)
    alpha → ∞ : IID uniform distribution
    """
    rng = np.random.default_rng(seed)
    classes = sorted(torch.unique(labels).tolist())
    client_indices = [[] for _ in range(n_clients)]
    for c in classes:
        idx_c = torch.where(labels == c)[0].numpy()
        rng.shuffle(idx_c)
        props  = rng.dirichlet(alpha * np.ones(n_clients))
        splits = (props * len(idx_c)).astype(int)
        splits[-1] = len(idx_c) - splits[:-1].sum()   # fix rounding
        start = 0
        for i, cnt in enumerate(splits):
            client_indices[i].extend(idx_c[start:start + cnt].tolist())
            start += cnt
    return [torch.tensor(sorted(ci), dtype=torch.long)
            for ci in client_indices]


# ── Client: compute local sums ────────────────────────────────

def client_compute_sums(latents_flat, labels, R, classes):
    """Clip latents and compute per-class sufficient statistics.

    This is all that leaves the client.  No raw latents or images
    are ever transmitted.

    Returns
    -------
    dict : class -> {"sum_z": (d,), "sum_zz": (d,d), "count": int}
    """
    clipped = clip_latents(latents_flat, R)
    sums = {}
    for c in classes:
        mask = labels == c
        if mask.sum() == 0:
            continue
        z_c = clipped[mask]
        sums[c] = {
            "sum_z":  z_c.sum(0),       # (d,)   sum of clipped latents
            "sum_zz": z_c.T @ z_c,      # (d,d)  sum of outer products
            "count":  int(mask.sum()),
        }
    return sums


# ── Secure aggregation (simulated) ───────────────────────────

def secure_aggregate(client_stats_list, classes):
    """Simulate SecAgg: sum per-class statistics across clients.

    In a real deployment, cryptographic secret sharing ensures the
    server only observes the totals -- never individual contributions.
    The simulation here is mathematically equivalent.

    Returns
    -------
    dict : class -> {"sum_z": (d,), "sum_zz": (d,d), "count": int}
    """
    global_sums = {}
    for c in classes:
        contribs = [s[c] for s in client_stats_list if c in s]
        if not contribs:
            continue
        global_sums[c] = {
            "sum_z":  sum(s["sum_z"]  for s in contribs),
            "sum_zz": sum(s["sum_zz"] for s in contribs),
            "count":  sum(s["count"]  for s in contribs),
        }
    return global_sums


# ── Server: DP noise injection ────────────────────────────────

def dp_from_sums(global_sums, epsilon, delta, R):
    """Add calibrated DP noise to globally aggregated sums.

    Adjacency: add/remove one record from the full dataset.
    Sensitivity of the sum queries:
        Delta_z  = R     (one clipped record, ||z||_2 <= R)
        Delta_zz = R^2   (one outer product, ||zz^T||_F = ||z||^2 <= R^2)

    After dividing sums by n_c the effective noise on the mean is
    sigma_z / n_c = _sigma(R, eps/2, delta/2) / n_c
                 = _sigma(R/n_c, eps/2, delta/2)
    which is *identical* to the centralized calibration in main.py.

    Post-processing (no additional privacy cost):
        mean      = noisy_sum_z  / n
        2nd mom   = noisy_sum_zz / n
        cov       = 2nd_mom - mean outer mean  +  bias correction
        bias corr = (sigma_z / n)^2 * I        [noise variance in mean]
        PSD proj  = clamp eigenvalues >= tau

    Returns
    -------
    stats    : dict  class -> {"mean": (d,), "cov": (d,d), "n": int}
               Compatible with dp.sample.
    sigma_z  : float  noise std added to sum_z
    sigma_zz : float  noise std added to sum_zz entries
    """
    # sensitivity = R / R^2 (add-remove); budget split eps/2 per query
    sigma_z  = _sigma(R,      epsilon / 2, delta / 2)
    sigma_zz = _sigma(R ** 2, epsilon / 2, delta / 2)

    stats = {}
    for c, g in global_sums.items():
        n = g["count"]
        d = g["sum_z"].shape[0]

        # add DP noise to the aggregated sums (server-side, after SecAgg)
        noisy_sum_z  = g["sum_z"]  + torch.randn(d) * sigma_z
        noise_mat    = torch.randn(d, d) * sigma_zz
        noisy_sum_zz = g["sum_zz"] + (noise_mat + noise_mat.T) / 2

        # recover mean and second moment
        mean       = noisy_sum_z  / n
        second_mom = noisy_sum_zz / n

        # covariance = E[zz^T] - mu mu^T  +  bias correction
        # bias: E[mu_noisy mu_noisy^T] = mu mu^T + (sigma_z/n)^2 I
        cov = second_mom - mean.outer(mean) + (sigma_z / n) ** 2 * torch.eye(d)

        stats[c] = {"mean": mean, "cov": _make_psd(cov), "n": n}

    return stats, sigma_z, sigma_zz


# ── Classifier ────────────────────────────────────────────────

def _make_resnet18(num_classes):
    import torch.nn as nn
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def train_and_eval(syn_x, syn_y, te_x, te_y, K, device, epochs=200):
    model = _make_resnet18(K).to(device)
    syn_x_norm = syn_x * 2 - 1
    te_x_norm  = te_x  * 2 - 1
    train_ld = DataLoader(TensorDataset(syn_x_norm, syn_y),
                          batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True)
    test_ld  = DataLoader(TensorDataset(te_x_norm,  te_y),
                          batch_size=128, shuffle=False,
                          num_workers=4, pin_memory=True)
    opt   = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                             weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=60, gamma=0.2)
    best  = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            mask = torch.rand(len(x), 1, 1, 1, device=device) < 0.5
            x = torch.where(mask, x.flip(-1), x)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()
        if (ep + 1) % 40 == 0 or ep == epochs - 1:
            model.eval()
            correct = sum(
                (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
                for x, y in test_ld)
            acc = 100 * correct / len(te_y)
            best = max(best, acc)
            print(f"    epoch {ep+1:>3}: {acc:.1f}%  (best {best:.1f}%)")
    return best


# ── Visualization ─────────────────────────────────────────────

def vis_partition(client_indices, labels, K, n_clients, out_dir, alpha):
    """Heatmap of class proportions per client."""
    matrix = np.zeros((n_clients, K))
    for i, idx in enumerate(client_indices):
        for c in range(K):
            matrix[i, c] = (labels[idx] == c).sum().item()
    matrix /= matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(max(6, K * 0.28), max(3, n_clients * 0.30)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Class index")
    ax.set_ylabel("Client index")
    ax.set_title(f"Data distribution  (α={alpha},  {n_clients} clients)")
    plt.colorbar(im, ax=ax, label="Fraction of client data")
    plt.tight_layout()
    path = os.path.join(out_dir, "client_distribution.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def vis_synthetic(syn_x, syn_y, K, out_dir, dataset, n_show=8):
    """Grid of synthetic images (first 10 classes)."""
    class_names = EUROSAT_CLASSES if dataset == "eurosat" else \
        [str(i) for i in range(K)]
    n_rows = min(K, 10)
    fig, axes = plt.subplots(n_rows, n_show, figsize=(n_show * 1.3, n_rows * 1.4))
    fig.suptitle(f"DP-Synthetic  ({dataset.upper()},  {n_clients_str(out_dir)})",
                 fontsize=13)
    for r in range(n_rows):
        syn_c = syn_x[syn_y == r]
        for col in range(n_show):
            ax = axes[r, col] if n_rows > 1 else axes[col]
            ax.imshow(syn_c[col].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(class_names[r], fontsize=7, rotation=0,
                              labelpad=50, va="center")
    plt.tight_layout()
    path = os.path.join(out_dir, "synthetic_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def n_clients_str(out_dir):
    """Read num_clients from results.json if available, else return ''."""
    p = os.path.join(out_dir, "results.json")
    if os.path.exists(p):
        d = json.load(open(p))
        return f"{d.get('num_clients','?')} clients, α={d.get('alpha','?')}"
    return ""


# ── Main ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    choices=["cifar100", "eurosat"], default="eurosat")
    p.add_argument("--data-dir",   default="./data")
    p.add_argument("--epsilon",    type=float, default=10.0)
    p.add_argument("--delta",      type=float, default=1e-5)
    p.add_argument("--clip-norm",  type=float, default=34.0)
    p.add_argument("--num-clients", type=int,  default=10)
    p.add_argument("--alpha",      type=float, default=0.5,
                   help="Dirichlet concentration: 0.1=very non-IID, 100=IID")
    p.add_argument("--samples-per-class", type=int, default=0)
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--dcae-model", default="dc-ae-f32c32-in-1.0")
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--output-dir", default="./output_fed")
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── 1. Data ───────────────────────────────────────────────
    print(f"[1/6] Loading {args.dataset} ...")
    tr_x, tr_y, te_x, te_y, K = load_dataset(args.dataset, args.data_dir)
    n_per_class = len(tr_y) // K
    spc         = args.samples_per_class or n_per_class
    classes     = sorted(tr_y.unique().tolist())
    print(f"  train: {tuple(tr_x.shape)},  K={K},  n/class~{n_per_class}")

    # ── 2. Partition ──────────────────────────────────────────
    print(f"\n[2/6] Partitioning → {args.num_clients} clients  (α={args.alpha}) ...")
    client_idx = partition_dirichlet(tr_y, args.num_clients, args.alpha, args.seed)
    sizes = [len(idx) for idx in client_idx]
    print(f"  sizes: min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.0f}, std={np.std(sizes):.0f}")
    vis_partition(client_idx, tr_y, K, args.num_clients,
                  args.output_dir, args.alpha)

    # ── 3. Encode ─────────────────────────────────────────────
    print(f"\n[3/6] Encoding with DC-AE ...")
    encoder, decoder = load_dcae(args.dcae_model, device)
    latents   = encode_all(encoder, tr_x, device=device)
    lat_shape = tuple(latents.shape[1:])
    flat      = latents.view(len(latents), -1)
    d         = flat.shape[1]
    norms     = flat.norm(dim=1)
    print(f"  latent: {lat_shape} → d={d}")
    print(f"  norms: mean={norms.mean():.1f},  "
          f"p90={norms.quantile(.9):.1f},  max={norms.max():.1f}")

    # ── 4. Clients compute sums  +  SecAgg ────────────────────
    R = args.clip_norm
    print(f"\n[4/6] Clients compute local sums  (R={R}) ...")
    print(f"  Each client sends only (sum_z, sum_zz, count) per class.")
    print(f"  No raw latents or images leave the client.")

    client_stats = []
    for i, idx in enumerate(client_idx):
        loc = client_compute_sums(flat[idx], tr_y[idx], R, classes)
        client_stats.append(loc)
        if i < 3 or i == args.num_clients - 1:
            print(f"  client {i:>3}: {len(idx):>5} samples, "
                  f"{len(loc)}/{K} classes present")

    print(f"  SecAgg: summing {args.num_clients} contributions ...")
    global_sums = secure_aggregate(client_stats, classes)

    # verify: aggregated sum == direct sum over full dataset
    # (floating-point accumulation over many clients may produce small diffs)
    clipped_all = clip_latents(flat, R)
    for c in classes[:3]:
        direct_sum = clipped_all[tr_y == c].sum(0)
        agg_sum    = global_sums[c]["sum_z"]
        rel_err = ((direct_sum - agg_sum).abs() /
                   direct_sum.abs().clamp(min=1e-6)).max().item()
        assert rel_err < 1e-3, f"SecAgg mismatch class {c}: rel_err={rel_err:.2e}"
    print(f"  ✓ Aggregated sums == direct computation (verified for 3 classes)")

    # ── 5. DP noise + synthesis ───────────────────────────────
    print(f"\n[5/6] DP noise injection + synthesis  (ε={args.epsilon}, δ={args.delta}) ...")
    stats, sigma_z, sigma_zz = dp_from_sums(global_sums, args.epsilon,
                                             args.delta, R)
    n_min = min(s["n"] for s in stats.values())
    print(f"  Sensitivity  Δ_z  = R   = {R:.1f}")
    print(f"  Sensitivity  Δ_zz = R²  = {R**2:.1f}")
    print(f"  Noise scale  σ_z        = {sigma_z:.4f}  (on sum)")
    print(f"  Noise scale  σ_zz       = {sigma_zz:.4f}  (on sum)")
    print(f"  Eff. noise on mean       = σ_z/n_min = {sigma_z/n_min:.6f}")
    print(f"  (Identical to centralised; independent of #clients and α)")

    print(f"\n  Generating {spc}×{K} = {spc*K} synthetic images ...")
    syn_z, syn_y = sample(stats, spc)
    syn_x = decode_all(decoder, syn_z, lat_shape, device=device)
    print(f"  synthetic: {tuple(syn_x.shape)}")

    torch.save({"images": syn_x, "labels": syn_y},
               os.path.join(args.output_dir, "synthetic.pt"))

    # save results.json now so vis_synthetic can read n_clients
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"dataset": args.dataset, "epsilon": args.epsilon,
                   "delta": args.delta, "clip_norm": R,
                   "num_clients": args.num_clients, "alpha": args.alpha,
                   "K": K, "n_min": n_min,
                   "sigma_z": sigma_z, "sigma_zz": sigma_zz}, f, indent=2)

    vis_synthetic(syn_x, syn_y, K, args.output_dir, args.dataset)

    del encoder, decoder
    torch.cuda.empty_cache()

    # ── 6. Train + evaluate ───────────────────────────────────
    print(f"\n[6/6] Training ResNet-18 ({args.epochs} epochs) ...")
    acc = train_and_eval(syn_x, syn_y, te_x, te_y, K, device, args.epochs)

    print(f"\n{'=' * 65}")
    print(f"  dataset          : {args.dataset}")
    print(f"  DP               : (ε={args.epsilon}, δ={args.delta})-DP  [add/remove]")
    print(f"  clip norm R      : {R}")
    print(f"  clients / α      : {args.num_clients} / {args.alpha}")
    print(f"  accuracy         : {acc:.1f}%")
    print()
    print(f"  Because the aggregated sums are identical for any partition")
    print(f"  of the same dataset, changing #clients or α produces the")
    print(f"  same DP statistics (up to random noise seed).")
    print(f"{'=' * 65}")

    # update results.json with accuracy
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"dataset": args.dataset, "epsilon": args.epsilon,
                   "delta": args.delta, "clip_norm": R,
                   "num_clients": args.num_clients, "alpha": args.alpha,
                   "accuracy": acc, "K": K, "n_min": n_min,
                   "sigma_z": sigma_z, "sigma_zz": sigma_zz,
                   "sensitivity_z": R, "sensitivity_zz": R ** 2}, f, indent=2)


if __name__ == "__main__":
    main()
