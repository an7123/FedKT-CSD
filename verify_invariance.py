#!/usr/bin/env python3
"""
Verify that the federated pipeline produces identical DP statistics
regardless of the number of clients and heterogeneity (alpha).

Mathematical claim
------------------
For any partition of the dataset into N clients:

    sum_i  sum_{j in client_i, class c}  clip(z_ij)
        == sum_{all j in class c}  clip(z_ij)

Because sums are additive, the aggregated sufficient statistics are
identical to the centralised computation.  With the same DP noise seed,
the recovered (mean, cov) per class are therefore bit-for-bit equal
across any (N_clients, alpha) configuration.

This script verifies that claim cheaply -- no classifier training needed.

Usage
-----
  python verify_invariance.py --dataset eurosat
  python verify_invariance.py --dataset cifar100 --clip-norm 44
"""

import argparse
import json
import os
import zipfile
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from codec import load_dcae
from dp import _sigma, _make_psd, clip_latents
from fed import (
    load_dataset, encode_all,
    partition_dirichlet, client_compute_sums, secure_aggregate, dp_from_sums,
)


def dp_from_sums_seeded(global_sums, epsilon, delta, R, seed):
    """dp_from_sums with a fixed noise seed for reproducible comparison."""
    torch.manual_seed(seed)
    return dp_from_sums(global_sums, epsilon, delta, R)


def stats_distance(s1, s2, classes):
    """Return max absolute difference in (mean, cov) across all classes."""
    max_mean_diff = 0.0
    max_cov_diff  = 0.0
    for c in classes:
        mean_diff = (s1[c]["mean"] - s2[c]["mean"]).abs().max().item()
        cov_diff  = (s1[c]["cov"]  - s2[c]["cov"] ).abs().max().item()
        max_mean_diff = max(max_mean_diff, mean_diff)
        max_cov_diff  = max(max_cov_diff,  cov_diff)
    return max_mean_diff, max_cov_diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   choices=["eurosat", "cifar100"], default="eurosat")
    p.add_argument("--epsilon",   type=float, default=10.0)
    p.add_argument("--delta",     type=float, default=1e-5)
    p.add_argument("--clip-norm", type=float, default=34.0)
    p.add_argument("--dcae-model", default="dc-ae-f32c32-in-1.0")
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--seed",      type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    R      = args.clip_norm

    # ── Load + encode (done once) ─────────────────────────────
    print(f"Loading {args.dataset} and encoding with DC-AE ...")
    tr_x, tr_y, _, _, K = load_dataset(args.dataset)
    encoder, _ = load_dcae(args.dcae_model, device)

    @torch.no_grad()
    def encode_all_local(imgs, bs=256):
        return torch.cat([encoder(imgs[i:i+bs].to(device)).cpu()
                          for i in range(0, len(imgs), bs)])

    latents = encode_all_local(tr_x)
    flat    = latents.view(len(latents), -1)
    classes = sorted(tr_y.unique().tolist())
    del encoder
    torch.cuda.empty_cache()

    print(f"  d={flat.shape[1]},  N={len(flat)},  K={K}")

    # ── Baseline: direct (centralised) aggregation ────────────
    print(f"\nBaseline: direct sum over full dataset ...")
    clipped = clip_latents(flat, R)
    baseline_global = {}
    for c in classes:
        z_c = clipped[tr_y == c]
        baseline_global[c] = {
            "sum_z":  z_c.sum(0),
            "sum_zz": z_c.T @ z_c,
            "count":  len(z_c),
        }
    baseline_stats, _, _ = dp_from_sums_seeded(
        baseline_global, args.epsilon, args.delta, R, seed=args.seed)
    print(f"  Done.  n_min = {min(g['count'] for g in baseline_global.values())}")

    # ── Sweep over (N_clients, alpha) ─────────────────────────
    configs = [
        (1,    100.0),   # single client (IID, trivial case)
        (5,    100.0),   # 5 clients, IID
        (10,   1.0),     # 10 clients, mild heterogeneity
        (10,   0.5),     # 10 clients, moderate heterogeneity
        (10,   0.1),     # 10 clients, high heterogeneity
        (20,   0.5),     # 20 clients, moderate
        (20,   0.1),     # 20 clients, high heterogeneity
        (50,   0.5),     # 50 clients
        (50,   0.1),     # 50 clients, high heterogeneity
        (100,  0.5),     # 100 clients
        (100,  0.1),     # 100 clients, high heterogeneity
    ]

    print(f"\n{'N':>5}  {'α':>6}  {'sum_err(max_rel)':>17}  "
          f"{'Δmean':>10}  {'Δcov':>10}  {'status':>8}")
    print("-" * 65)

    all_passed = True
    for n_clients, alpha in configs:
        # partition
        client_idx = partition_dirichlet(tr_y, n_clients, alpha, seed=42)

        # clients compute local sums
        client_stats = [
            client_compute_sums(flat[idx], tr_y[idx], R, classes)
            for idx in client_idx
        ]

        # SecAgg
        global_sums = secure_aggregate(client_stats, classes)

        # check sums match baseline (pure math, no noise)
        max_rel_err = 0.0
        for c in classes:
            direct = baseline_global[c]["sum_z"]
            agg    = global_sums[c]["sum_z"]
            rel    = ((direct - agg).abs() /
                      direct.abs().clamp(min=1e-6)).max().item()
            max_rel_err = max(max_rel_err, rel)

        # apply DP noise with same seed → should give identical stats
        fed_stats, _, _ = dp_from_sums_seeded(
            global_sums, args.epsilon, args.delta, R, seed=args.seed)

        d_mean, d_cov = stats_distance(baseline_stats, fed_stats, classes)

        ok = max_rel_err < 1e-3 and d_mean < 1e-4 and d_cov < 1e-2
        all_passed = all_passed and ok
        status = "PASS" if ok else "FAIL"

        print(f"{n_clients:>5}  {alpha:>6.1f}  {max_rel_err:>17.2e}  "
              f"{d_mean:>10.2e}  {d_cov:>10.2e}  {status:>8}")

    print("-" * 65)
    print(f"\n{'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print()
    print("Interpretation")
    print("--------------")
    print("  sum_err : relative error in aggregated sums vs direct computation")
    print("            (should be ~0, only floating-point rounding)")
    print("  Δmean   : max abs difference in DP-protected class means")
    print("  Δcov    : max abs difference in DP-protected covariance matrices")
    print("  Both Δ should be exactly 0 (same noise seed = same noise vectors).")
    print("  Any residual is pure float32 rounding from the partition loop.")


if __name__ == "__main__":
    main()
