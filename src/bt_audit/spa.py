"""Hansen's test for Superior Predictive Ability (SPA), with White's Reality Check alongside.

References
----------
Hansen, P. R. (2005). A Test for Superior Predictive Ability. Journal of Business & Economic
    Statistics 23(4), 365-380.
White, H. (2000). A Reality Check for Data Snooping. Econometrica 68(5), 1097-1126.
Politis, D. N. and Romano, J. P. (1994). The Stationary Bootstrap. JASA 89(428).

Setup used here
---------------
Loss is the negative period return, and the benchmark is a zero return. The loss differential
of variant k at period t is therefore

    d_{k,t} = L_{0,t} - L_{k,t} = 0 - (-r_{k,t}) = r_{k,t},

so a positive mean d_k means variant k beat the benchmark. With active returns (strategy minus
benchmark) as input, a zero active return already is the benchmark.

The null is that no variant beats the benchmark: max_k E[d_k] <= 0.

    T_SPA = max(0, max_k sqrt(n) dbar_k / omega_k)                       (Hansen eq. 5)

omega_k^2 is the stationary-bootstrap variance estimator of sqrt(n) dbar_k (Hansen 2005,
section 2.3, after Politis and Romano 1994):

    omega^2 = gamma_0 + 2 sum_{i=1}^{n-1} kappa(n, i) gamma_i,
    kappa(n, i) = ((n - i) / n) (1 - q)^i + (i / n) (1 - q)^(n - i),  q = 1 / mean block.

Bootstrap: resample periods with the stationary bootstrap (the same index draw for every
variant, which keeps their correlation), and recentre each variant's bootstrap mean by g(dbar_k):

    g_l(x) = max(x, 0)                                       -> SPA_l (liberal, lower bound)
    g_c(x) = x * 1{x >= -sqrt(omega^2 / n * 2 log log n)}    -> SPA_c (consistent, the headline)
    g_u(x) = x                                               -> SPA_u (conservative, upper bound)

    T*_b = max(0, max_k sqrt(n) (dbar*_{k,b} - g(dbar_k)) / omega_k),  p = mean(T*_b > T_SPA).

SPA_c drops variants that are clearly worse than the benchmark from the null distribution,
which is why a pile of bad variants cannot hide a good one. White's Reality Check is the
non-studentized, fully recentred version:

    V = max_k sqrt(n) dbar_k,   V*_b = max_k sqrt(n) (dbar*_{k,b} - dbar_k),  p = mean(V*_b > V).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .stress import stationary_bootstrap_indices


@dataclass(frozen=True)
class SPAResult:
    p_consistent: float  # SPA_c, the headline p-value
    p_lower: float  # SPA_l
    p_upper: float  # SPA_u
    p_reality_check: float  # White (2000)
    statistic: float  # T_SPA
    best_index: int  # variant with the largest studentized mean
    best_t: float  # its studentized mean sqrt(n) dbar / omega
    n_variants: int
    n_periods: int
    n_boot: int
    mean_block: float
    n_dropped_by_consistent: int  # variants SPA_c treats as clearly worse than the benchmark


def _autocov(x: np.ndarray) -> np.ndarray:
    """Autocovariances gamma_0..gamma_{n-1} of each column (divisor n), via FFT."""
    n = x.shape[0]
    c = x - x.mean(axis=0)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(c, n=size, axis=0)
    return np.fft.irfft(f * np.conj(f), n=size, axis=0)[:n] / n


def long_run_variance(d: np.ndarray, mean_block: float) -> np.ndarray:
    """Stationary-bootstrap variance of sqrt(n) * mean, per column (Hansen 2005, section 2.3)."""
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    n = d.shape[0]
    q = 1.0 / max(mean_block, 1.0)
    g = _autocov(d)
    i = np.arange(1, n, dtype=float)
    kappa = (n - i) / n * (1.0 - q) ** i + i / n * (1.0 - q) ** (n - i)
    return g[0] + 2.0 * (kappa[:, None] * g[1:]).sum(axis=0)


def spa(
    returns: np.ndarray,
    n_boot: int = 2000,
    mean_block: float | None = None,
    seed: int = 7,
    batch: int = 250,
) -> SPAResult:
    """Hansen SPA test of 'some variant beats a zero return'. returns is T x N (N >= 1)."""
    d = np.asarray(returns, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    d = d[np.all(np.isfinite(d), axis=1)]
    n, k = d.shape
    if n < 10:
        raise ValueError(f"need at least 10 complete rows, got {n}")
    keep = d.std(axis=0) > 0
    if not keep.any():
        raise ValueError("every variant has zero variance")
    cols = np.flatnonzero(keep)
    d = d[:, keep]
    if mean_block is None:
        # n^(1/3) is the usual rate for block bootstraps of a mean. sqrt(n), the sizing rule, made the
        # variance estimate omega too small here and the test rejected ~10% of the time at a 5% level
        # on independent zero-edge variants (simulated, T=1000, N=20); n^(1/3) holds ~5%.
        mean_block = n ** (1.0 / 3.0)

    dbar = d.mean(axis=0)
    omega = np.sqrt(np.maximum(long_run_variance(d, mean_block), 1e-300))
    rn = math.sqrt(n)
    tstat = rn * dbar / omega
    t_spa = max(float(tstat.max()), 0.0)
    v_rc = float(rn * dbar.max())

    thresh = np.sqrt(omega**2 / n * 2.0 * math.log(math.log(n)))
    g_l = np.maximum(dbar, 0.0)
    g_c = np.where(dbar >= -thresh, dbar, 0.0)
    g_u = dbar

    rng = np.random.default_rng(seed)
    hits = np.zeros(4)
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        idx = stationary_bootstrap_indices(n, b, mean_block, rng)
        # bootstrap means through index counts: (b x n) @ (n x k), never materialising b x n x k
        flat = (idx + (np.arange(b) * n)[:, None]).ravel()
        counts = np.bincount(flat, minlength=b * n).reshape(b, n)
        mstar = counts @ d / n  # b x k
        for j, g in enumerate((g_l, g_c, g_u)):
            ts = np.maximum((rn * (mstar - g) / omega).max(axis=1), 0.0)
            hits[j] += np.sum(ts > t_spa)
        hits[3] += np.sum((rn * (mstar - dbar)).max(axis=1) > v_rc)
        done += b
    p = hits / n_boot
    best = int(tstat.argmax())
    return SPAResult(
        p_consistent=float(p[1]),
        p_lower=float(p[0]),
        p_upper=float(p[2]),
        p_reality_check=float(p[3]),
        statistic=t_spa,
        best_index=int(cols[best]),
        best_t=float(tstat[best]),
        n_variants=int(k),
        n_periods=int(n),
        n_boot=int(n_boot),
        mean_block=float(mean_block),
        n_dropped_by_consistent=int(np.sum(dbar < -thresh)),
    )
