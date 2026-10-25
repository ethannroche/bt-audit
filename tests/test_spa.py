"""Hansen SPA checked by simulation, not by remembered numbers.

Tests use 300 bootstrap resamples per p-value (the report uses 2,000) so the suite stays
fast. That adds Monte Carlo noise of about +/-0.01 to each p-value near 0.05, which the
tolerances below allow for.
"""

import numpy as np
import pytest

from bt_audit import spa as spa_mod
from bt_audit.spa import long_run_variance, spa

N_BOOT = 300


def _null_rejection_rate(rho, runs=400, t=1000, n=20, seed=123):
    rng = np.random.default_rng(seed)
    rej = 0
    for s in range(runs):
        common = rng.normal(0, 0.01, (t, 1))
        x = np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.normal(0, 0.01, (t, n))
        rej += spa(x, n_boot=N_BOOT, seed=s).p_consistent <= 0.05
    return rej / runs


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_size_under_zero_edge(rho):
    """20 zero-edge variants, iid and with pairwise correlation 0.5: SPA_c rejects at 5% no more
    than about 8% of the time (400 runs, so one standard error is about 1.1 points)."""
    assert _null_rejection_rate(rho) <= 0.08


def test_power_one_real_edge_among_ten():
    """One variant with true per-period Sharpe 0.12 among 10, T = 2,000: rejected in >= 90% of runs."""
    rng = np.random.default_rng(5)
    runs, hits = 60, 0
    for s in range(runs):
        x = rng.normal(0, 0.01, (2000, 10))
        x[:, 3] += 0.0012
        hits += spa(x, n_boot=N_BOOT, seed=s).p_consistent <= 0.05
    assert hits / runs >= 0.90


def test_poor_variants_do_not_hide_a_winner():
    """The reason SPA exists (Hansen 2005, section 1): White's Reality Check recentres every variant
    at its own mean, so a pile of clearly losing variants inflates its null and its p-value.
    SPA_c drops them, so its p-value is at or below the Reality Check's and rejects far more often.
    """
    rng = np.random.default_rng(9)
    pc, rc = [], []
    for s in range(40):
        x = rng.normal(0, 0.01, (1000, 30))
        x[:, 0] += 0.0008
        x[:, 1:] -= 0.003
        r = spa(x, n_boot=N_BOOT, seed=s)
        pc.append(r.p_consistent)
        rc.append(r.p_reality_check)
        assert r.p_lower <= r.p_consistent <= r.p_upper
    pc, rc = np.array(pc), np.array(rc)
    assert np.all(pc <= rc + 1e-12)
    assert np.mean(pc <= 0.05) >= 0.6
    assert np.mean(rc <= 0.05) <= np.mean(pc <= 0.05) - 0.2


def test_ordering_and_determinism():
    x = np.random.default_rng(1).normal(0.0002, 0.01, (800, 8))
    a, b = spa(x, n_boot=N_BOOT, seed=3), spa(x, n_boot=N_BOOT, seed=3)
    assert a == b
    assert a.p_lower <= a.p_consistent <= a.p_upper


def test_long_run_variance_of_iid_is_the_variance():
    x = np.random.default_rng(2).normal(0, 2.0, (20000, 1))
    assert long_run_variance(x, 5.0)[0] == pytest.approx(4.0, rel=0.05)


def test_long_run_variance_sees_autocorrelation():
    """AR(1) with phi 0.5: the variance of sqrt(n) * mean is sigma^2 / (1 - phi)^2 = 4 x the innovations'."""
    rng = np.random.default_rng(3)
    e = rng.normal(0, 1, 40000)
    y = np.empty_like(e)
    y[0] = e[0]
    for i in range(1, e.size):
        y[i] = 0.5 * y[i - 1] + e[i]
    assert long_run_variance(y[:, None], 40000 ** (1 / 3))[0] == pytest.approx(4.0, rel=0.15)


def test_report_includes_spa_and_it_sets_the_grade():
    from bt_audit.report import run_audit, to_dict

    x = np.random.default_rng(0).normal(0, 0.01, (1000, 20))
    best = x[:, int((x.mean(0) / x.std(0)).argmax())]
    a = run_audit(best, x)
    c = next(c for c in a.checks if c.name == "Selection bias (Hansen SPA)")
    assert c.verdict in {"WARN", "FAIL"}
    assert "d_k,t = r_k,t" in " ".join(c.detail)
    d = to_dict(a)
    assert 0.0 <= d["spa"]["p_consistent"] <= 1.0
    assert d["spa"]["n_variants"] == 20  # the audited column was found among the trials, not added


def test_spa_skips_without_a_matrix():
    from bt_audit.report import run_audit

    r = np.random.default_rng(0).normal(0.0005, 0.01, 500)
    c = next(c for c in run_audit(r, n_trials=20).checks if c.name == "Selection bias (Hansen SPA)")
    assert c.verdict == "SKIP"


def test_spa_module_cites_hansen():
    assert "Hansen, P. R. (2005)" in spa_mod.__doc__
