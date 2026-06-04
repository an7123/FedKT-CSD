"""(epsilon, delta)-Differential Privacy via Gaussian mechanism.

Computes per-class sufficient statistics (mean and covariance) from
L2-clipped latent codes, adds calibrated Gaussian noise, and samples
synthetic latent codes from the protected distributions.

Privacy accounting
------------------
  Budget split : epsilon/2 for mean, epsilon/2 for second moment.
  Composition  : basic sequential within each class,
                 parallel across classes (disjoint data).
  Sensitivity  : R/n for mean, R^2/n for E[zz^T]  (add/remove-one).
  Post-proc    : bias correction, PSD projection, sampling, decoding.
  Calibration  : Analytic Gaussian mechanism (Balle & Wang, NeurIPS 2018).
"""

import math
import torch
from scipy.optimize import brentq
from scipy.stats import norm as _ndist


def _sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Analytic Gaussian mechanism (Balle & Wang, 2018).

    Finds the exact minimum sigma such that the Gaussian mechanism with
    L2-sensitivity `sensitivity` satisfies (epsilon, delta)-DP.
    Tighter than the standard bound and valid for all epsilon > 0.
    """
    if sensitivity == 0:
        return 0.0

    def _delta_of_sigma(sigma):
        a = sensitivity / (2 * sigma) - epsilon * sigma / sensitivity
        b = -sensitivity / (2 * sigma) - epsilon * sigma / sensitivity
        return _ndist.cdf(a) - math.exp(epsilon) * _ndist.cdf(b)

    # Upper bound: the (loose) standard calibration always over-estimates
    hi = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    # But the standard bound can be invalid at high eps — ensure bracket
    while _delta_of_sigma(hi) > delta:
        hi *= 2
    lo = 1e-16
    return brentq(lambda s: _delta_of_sigma(s) - delta, lo, hi, xtol=1e-12)


def _make_psd(M: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    """Project symmetric matrix onto PSD cone via eigenvalue clamping."""
    vals, vecs = torch.linalg.eigh(M)
    return vecs @ torch.diag(vals.clamp(min=floor)) @ vecs.T


# ── public API ────────────────────────────────────────────────────

def clip_latents(z: torch.Tensor, R: float) -> torch.Tensor:
    """L2-clip each row to norm <= R."""
    norms = z.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return z * (R / norms).clamp(max=1.0)


def dp_statistics(
    latents: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    delta: float,
    R: float,
) -> dict:
    """
    Per-class (epsilon, delta)-DP mean and covariance.

    Args
    ----
    latents : (N, d) clipped codes, ||z_i|| <= R.
    labels  : (N,)   integer class labels.
    epsilon : total privacy budget.
    delta   : failure probability.
    R       : L2 clipping threshold (data-independent).

    Returns
    -------
    dict : class -> {"mean": (d,), "cov": (d, d), "n": int}.
    """
    d = latents.shape[1]
    classes = sorted(torch.unique(labels).tolist())

    stats = {}
    for c in classes:
        z = latents[labels == c]
        n = len(z)

        # noise calibrated to eps/2, delta/2 per query
        s_mu  = _sigma(R / n,      epsilon / 2, delta / 2)
        s_cov = _sigma(R ** 2 / n, epsilon / 2, delta / 2)

        # noisy mean
        mean = z.mean(0) + torch.randn(d) * s_mu

        # noisy second moment (symmetric noise is post-processing)
        noise  = torch.randn(d, d) * s_cov
        second = z.T @ z / n + (noise + noise.T) / 2

        # covariance  =  E[zz^T] - E[z]E[z]^T  +  bias correction
        cov = second - mean.outer(mean) + s_mu ** 2 * torch.eye(d)

        stats[c] = {"mean": mean, "cov": _make_psd(cov), "n": n}
    return stats


def sample(stats: dict, n_per_class: int) -> tuple:
    """Sample latent codes from DP-protected per-class Gaussians."""
    all_z, all_y = [], []
    for c in sorted(stats):
        mu, cov = stats[c]["mean"], stats[c]["cov"]
        try:
            L = torch.linalg.cholesky(cov)
        except RuntimeError:
            L = torch.linalg.cholesky(cov + 1e-3 * torch.eye(len(mu)))
        z = mu.unsqueeze(0) + torch.randn(n_per_class, len(mu)) @ L.T
        all_z.append(z)
        all_y.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(all_z), torch.cat(all_y)
