"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation (CSCV).

Reference
---------
Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017). The Probability of
Backtest Overfitting. Journal of Computational Finance 20(4).

Procedure: split the T x N matrix of trial returns into S equal row blocks. For every way of
choosing S/2 blocks as in-sample (the rest out-of-sample), pick the trial with the best
in-sample Sharpe and record where it ranks out-of-sample. PBO is the share of splits in which
the in-sample winner lands at or below the out-of-sample median.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_splits: int
    n_trials: int
    logits: np.ndarray  # one lambda per split
    is_sharpe_best: np.ndarray  # in-sample Sharpe of the chosen trial, per split
    oos_sharpe_best: np.ndarray  # its out-of-sample Sharpe, per split
    prob_oos_loss: float  # share of splits where the chosen trial lost money out-of-sample
    rows_dropped: int  # oldest rows trimmed so the blocks are equal


def _sharpe_from_sums(s: np.ndarray, ss: np.ndarray, n: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Sharpe from sums of CENTRED returns (s, ss) plus the column means mu that were removed."""
    mean_c = s / n
    var = (ss - s * s / n) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = (mean_c + mu) / np.sqrt(var)
    return np.where(var > 1e-30, sr, 0.0)


def pbo(trials: np.ndarray, n_blocks: int = 16) -> PBOResult:
    m = np.asarray(trials, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("trials must be a T x N matrix with N >= 2")
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be an even number >= 2")
    m = m[np.all(np.isfinite(m), axis=1)]
    t, n = m.shape
    rows = (t // n_blocks) * n_blocks
    if rows < n_blocks * 2:
        raise ValueError(f"need at least {n_blocks * 2} complete rows, got {t}")
    dropped = t - rows
    m = m[dropped:]  # drop the OLDEST rows so the blocks are equal; recent data is kept
    # centre each column first: the one-pass variance below is exact on centred data and loses
    # precision through cancellation when |mean| >> sd on raw data
    mu = m.mean(axis=0)
    c = m - mu
    blocks = c.reshape(n_blocks, rows // n_blocks, n)
    b_sum = blocks.sum(axis=1)  # S x N
    b_ss = (blocks**2).sum(axis=1)  # S x N
    b_n = np.full(n_blocks, rows // n_blocks, dtype=float)

    combos = list(combinations(range(n_blocks), n_blocks // 2))
    ind = np.zeros((len(combos), n_blocks))
    for k, c in enumerate(combos):
        ind[k, list(c)] = 1.0
    oind = 1.0 - ind

    is_sr = _sharpe_from_sums(ind @ b_sum, ind @ b_ss, (ind @ b_n)[:, None], mu)
    oos_sr = _sharpe_from_sums(oind @ b_sum, oind @ b_ss, (oind @ b_n)[:, None], mu)

    best = is_sr.argmax(axis=1)
    k_idx = np.arange(len(combos))
    oos_rank = rankdata(oos_sr, axis=1)[k_idx, best]  # 1 = worst, n = best
    omega = oos_rank / (n + 1.0)
    logits = np.log(omega / (1.0 - omega))

    oos_best = oos_sr[k_idx, best]
    return PBOResult(
        pbo=float(np.mean(logits <= 0)),
        n_splits=len(combos),
        n_trials=n,
        logits=logits,
        is_sharpe_best=is_sr[k_idx, best],
        oos_sharpe_best=oos_best,
        prob_oos_loss=float(np.mean(oos_best < 0)),
        rows_dropped=int(dropped),
    )
