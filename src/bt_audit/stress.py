"""Cost and sizing stress tests.

Reference for the bootstrap
---------------------------
Politis, D. N. and Romano, J. P. (1994). The Stationary Bootstrap. Journal of the American
Statistical Association 89(428). Blocks of random (geometric) length keep the short-range
dependence of returns, which an i.i.d. shuffle would destroy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostResult:
    bps_grid: tuple[float, ...]
    sharpe_annual: tuple[float, ...]
    breakeven_bps: float  # cost per unit turnover at which mean return reaches zero


def cost_stress(
    returns: np.ndarray,
    turnover: float,
    periods_per_year: int = 252,
    bps_grid: tuple[float, ...] = (0, 5, 10, 20, 50),
) -> CostResult:
    """Returns are treated as GROSS of costs. Cost per period = turnover * bps / 10,000."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if turnover <= 0:
        raise ValueError("turnover must be > 0")
    out = []
    for bps in bps_grid:
        net = r - turnover * bps / 1e4
        sd = net.std(ddof=1)
        out.append(float(net.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0)
    mean = float(r.mean())
    breakeven = mean / turnover * 1e4 if mean > 0 else 0.0
    return CostResult(tuple(float(b) for b in bps_grid), tuple(out), float(breakeven))


def stationary_bootstrap_indices(
    t: int, n_paths: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    p_jump = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_paths, t), dtype=np.int64)
    idx[:, 0] = rng.integers(0, t, n_paths)
    for k in range(1, t):
        jump = rng.random(n_paths) < p_jump
        idx[:, k] = np.where(jump, rng.integers(0, t, n_paths), (idx[:, k - 1] + 1) % t)
    return idx


@dataclass(frozen=True)
class SizingRow:
    leverage: float
    median_max_drawdown: float
    p95_max_drawdown: float
    prob_drawdown_50: float  # P(max drawdown >= 50%)
    prob_ruin: float  # P(wealth hits zero)


def sizing_stress(
    returns: np.ndarray,
    leverages: tuple[float, ...] = (1.0, 2.0, 3.0),
    n_paths: int = 5000,
    seed: int = 7,
) -> list[SizingRow]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    t = r.size
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(t, n_paths, math.sqrt(t), rng)
    paths = r[idx]
    rows = []
    for lev in leverages:
        growth = 1.0 + lev * paths
        ruined_step = growth <= 0
        growth = np.where(ruined_step, 0.0, growth)
        wealth = np.cumprod(growth, axis=1)
        peak = np.maximum.accumulate(np.concatenate([np.ones((n_paths, 1)), wealth], axis=1), axis=1)[:, 1:]
        dd = 1.0 - wealth / peak
        mdd = dd.max(axis=1)
        rows.append(
            SizingRow(
                leverage=float(lev),
                median_max_drawdown=float(np.median(mdd)),
                p95_max_drawdown=float(np.quantile(mdd, 0.95)),
                prob_drawdown_50=float(np.mean(mdd >= 0.5)),
                prob_ruin=float(np.mean(ruined_step.any(axis=1))),
            )
        )
    return rows
