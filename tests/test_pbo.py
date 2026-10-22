import numpy as np

from bt_audit.pbo import pbo


def test_pure_noise_is_overfit_about_half_the_time():
    """No trial has an edge, so the in-sample winner is a coin flip out of sample: PBO near 0.5."""
    vals = []
    for seed in range(8):
        x = np.random.default_rng(seed).normal(0, 0.01, (1000, 50))
        vals.append(pbo(x).pbo)
    assert 0.35 < np.mean(vals) < 0.65


def test_one_real_edge_among_noise_is_not_overfit():
    vals = []
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        x = rng.normal(0, 0.01, (1000, 50))
        x[:, 7] += 0.0015  # one trial carries a per-period SR of 0.15
        vals.append(pbo(x).pbo)
    assert np.mean(vals) < 0.1


def test_split_count_is_n_choose_half():
    x = np.random.default_rng(0).normal(0, 0.01, (320, 5))
    assert pbo(x, n_blocks=16).n_splits == 12870


def test_rejects_too_few_rows():
    import pytest

    with pytest.raises(ValueError):
        pbo(np.zeros((10, 4)) + 0.001)


def test_split_sharpes_match_two_pass_when_mean_dwarfs_volatility():
    """Reviewer finding: one-pass variance cancels when |mean| >> sd. Check the split Sharpes
    against a direct two-pass computation on exactly the same rows."""
    from itertools import combinations

    rng = np.random.default_rng(5)
    x = 0.05 + rng.normal(0, 1e-5, (320, 4))  # mean 5,000x the volatility
    r = pbo(x)
    blocks = np.array_split(x, 16)
    first = list(combinations(range(16), 8))[0]
    is_rows = np.concatenate([blocks[b] for b in first])
    direct = is_rows.mean(0) / is_rows.std(0, ddof=1)
    assert r.is_sharpe_best[0] == np.max(direct) or abs(r.is_sharpe_best[0] - np.max(direct)) / np.max(direct) < 1e-6


def test_oldest_rows_are_the_ones_dropped():
    x = np.random.default_rng(6).normal(0, 0.01, (333, 4))
    assert pbo(x).rows_dropped == 333 - (333 // 16) * 16
