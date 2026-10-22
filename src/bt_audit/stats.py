"""Sharpe-ratio significance tests.

All Sharpe ratios here are PER PERIOD (not annualised) unless a name says otherwise, because the
published formulas are stated per period. Annualise only for display.

References
----------
Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
    Journal of Risk 15(2). (PSR, MinTRL)
Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality. Journal of Portfolio Management 40(5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class Moments:
    n: int
    sharpe: float  # per period
    skew: float  # gamma_3
    kurtosis: float  # gamma_4, NOT excess (normal = 3)


def moments(returns: np.ndarray) -> Moments:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        raise ValueError("need at least 3 finite returns")
    sd = r.std(ddof=1)
    if sd == 0:
        raise ValueError("returns have zero variance")
    return Moments(
        n=int(r.size),
        sharpe=float(r.mean() / sd),
        skew=float(st.skew(r, bias=False)),
        kurtosis=float(st.kurtosis(r, fisher=False, bias=False)),
    )


def _sr_std(m: Moments) -> float:
    """Standard error term of the Sharpe estimate under non-normal returns (Bailey & LdP 2012)."""
    var = 1.0 - m.skew * m.sharpe + (m.kurtosis - 1.0) / 4.0 * m.sharpe**2
    # var can go negative for absurd skew/kurtosis combos; floor it so the test stays defined
    return math.sqrt(max(var, 1e-12))


def psr(m: Moments, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_benchmark). Both per period."""
    z = (m.sharpe - sr_benchmark) * math.sqrt(m.n - 1) / _sr_std(m)
    return float(st.norm.cdf(z))


def min_track_record_length(m: Moments, sr_benchmark: float = 0.0, alpha: float = 0.05) -> float:
    """Periods needed for PSR >= 1 - alpha. Infinite when the observed SR does not beat the benchmark."""
    if m.sharpe <= sr_benchmark:
        return math.inf
    z = st.norm.ppf(1.0 - alpha)
    var = 1.0 - m.skew * m.sharpe + (m.kurtosis - 1.0) / 4.0 * m.sharpe**2
    return float(1.0 + max(var, 1e-12) * (z / (m.sharpe - sr_benchmark)) ** 2)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among n_trials zero-edge strategies (Bailey & LdP 2014, eq. 6).

    sr_variance is the cross-sectional variance of the trials' per-period Sharpe estimates.
    """
    if n_trials < 2:
        return 0.0
    a = st.norm.ppf(1.0 - 1.0 / n_trials)
    b = st.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def dsr(m: Moments, n_trials: int, sr_variance: float | None = None) -> float:
    """Deflated Sharpe Ratio: PSR against the Sharpe you would expect from the best of n_trials
    strategies with no edge.

    If sr_variance is unknown (only N is known), fall back to the variance of a zero-edge Sharpe
    estimate over this track record, 1/(T-1). That is the null the deflation is testing.
    """
    if sr_variance is None:
        sr_variance = 1.0 / (m.n - 1)
    return psr(m, expected_max_sharpe(n_trials, sr_variance))


def mean_pairwise_correlation(trials: np.ndarray) -> float:
    """Mean off-diagonal correlation of the trials' returns, floored at 0 (negative mean
    correlation is treated as independence, the stricter case)."""
    m = np.asarray(trials, dtype=float)
    m = m[np.all(np.isfinite(m), axis=1)]
    m = m[:, m.std(axis=0, ddof=1) > 0]
    k = m.shape[1]
    if k < 2:
        return 0.0
    c = np.corrcoef(m, rowvar=False)
    return float(min(max((c.sum() - k) / (k * (k - 1)), 0.0), 0.999))


def annualise(sr_per_period: float, periods_per_year: int) -> float:
    return sr_per_period * math.sqrt(periods_per_year)
