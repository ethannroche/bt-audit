"""The formulas are checked against simulation, not against remembered worked examples:
under a known null, a correct test has a known behaviour."""

import math

import numpy as np
import pytest

from bt_audit.stats import (
    dsr,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    psr,
)


def test_moments_normal_kurtosis_is_three():
    r = np.random.default_rng(0).normal(0, 0.01, 200_000)
    m = moments(r)
    assert abs(m.kurtosis - 3.0) < 0.05
    assert abs(m.skew) < 0.05


def test_psr_is_uniform_under_the_null():
    """True SR = 0: P(PSR > 0.95) should be about 5% (a correctly sized 5% test)."""
    rng = np.random.default_rng(1)
    hits = sum(psr(moments(rng.normal(0, 0.01, 500))) > 0.95 for _ in range(4000))
    assert 0.04 < hits / 4000 < 0.06


def test_psr_detects_a_real_edge():
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, 1000)  # per-period SR 0.1, about 1.6 annualised
    assert psr(moments(r)) > 0.99


def test_expected_max_sharpe_matches_simulation():
    """E[max SR of N zero-edge strategies] from the closed form vs brute force."""
    rng = np.random.default_rng(3)
    n_trials, t = 50, 500
    sims = []
    for _ in range(400):
        x = rng.normal(0, 0.01, (t, n_trials))
        sr = x.mean(axis=0) / x.std(axis=0, ddof=1)
        sims.append(sr.max())
    closed = expected_max_sharpe(n_trials, 1.0 / (t - 1))
    assert abs(closed - np.mean(sims)) / np.mean(sims) < 0.05


def test_dsr_kills_the_best_of_many_noise_trials():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.01, (750, 100))
    sr = x.mean(axis=0) / x.std(axis=0, ddof=1)
    best = x[:, sr.argmax()]
    m = moments(best)
    assert psr(m) > 0.95  # naive test is fooled
    assert dsr(m, n_trials=100, sr_variance=float(sr.var(ddof=1))) < 0.5  # deflated test is not


def test_min_track_record_length_is_consistent_with_psr():
    rng = np.random.default_rng(5)
    m = moments(rng.normal(0.0008, 0.01, 2000))
    mintrl = min_track_record_length(m)
    # at exactly MinTRL periods, PSR should sit at the 95% boundary
    m_at = type(m)(n=int(round(mintrl)), sharpe=m.sharpe, skew=m.skew, kurtosis=m.kurtosis)
    assert psr(m_at) == pytest.approx(0.95, abs=0.01)


def test_min_track_record_length_infinite_without_edge():
    m = moments(np.random.default_rng(6).normal(-0.001, 0.01, 500))
    assert math.isinf(min_track_record_length(m))


def test_negative_skew_widens_uncertainty():
    """Fat left tail (a short-vol style strategy) must lower PSR at the same Sharpe."""
    from bt_audit.stats import Moments

    base = Moments(n=500, sharpe=0.08, skew=0.0, kurtosis=3.0)
    crashy = Moments(n=500, sharpe=0.08, skew=-2.0, kurtosis=12.0)
    assert psr(crashy) < psr(base)


def test_mean_pairwise_correlation():
    from bt_audit.stats import mean_pairwise_correlation

    rng = np.random.default_rng(9)
    common = rng.normal(0, 0.01, (4000, 1))
    x = np.sqrt(0.5) * common + np.sqrt(0.5) * rng.normal(0, 0.01, (4000, 20))
    assert abs(mean_pairwise_correlation(x) - 0.5) < 0.05
    assert mean_pairwise_correlation(rng.normal(0, 0.01, (4000, 20))) < 0.02


def _null_pass_rate(rho, n=100, t=1000, runs=300, seed=11):
    """How often the best of n zero-edge trials passes the deflated test as report.py runs it."""
    from bt_audit.stats import mean_pairwise_correlation

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(runs):
        common = rng.normal(0, 0.01, (t, 1))
        x = np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.normal(0, 0.01, (t, n))
        sr = x.mean(0) / x.std(0, ddof=1)
        m = moments(x[:, sr.argmax()])
        v = (1 - mean_pairwise_correlation(x)) / (m.n - 1)
        hits += dsr(m, n, v) > 0.95
    return hits / runs


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_deflated_test_is_not_anti_conservative(rho):
    """Regression for the reviewer's finding: zero-edge trials must pass at or below the nominal 5%."""
    assert _null_pass_rate(rho) <= 0.07
