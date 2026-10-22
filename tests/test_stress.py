import numpy as np
import pytest

from bt_audit.stress import cost_stress, sizing_stress


def test_breakeven_cost_is_where_mean_hits_zero():
    r = np.random.default_rng(0).normal(0.0005, 0.01, 2000)
    res = cost_stress(r, turnover=0.5)
    net_mean = r.mean() - 0.5 * res.breakeven_bps / 1e4
    assert net_mean == pytest.approx(0.0, abs=1e-12)


def test_sharpe_falls_monotonically_with_cost():
    r = np.random.default_rng(1).normal(0.0005, 0.01, 2000)
    s = cost_stress(r, turnover=1.0).sharpe_annual
    assert all(a > b for a, b in zip(s, s[1:]))


def test_leverage_raises_drawdown_risk():
    r = np.random.default_rng(2).normal(0.0004, 0.012, 1000)
    rows = sizing_stress(r, n_paths=1000)
    p = [row.prob_drawdown_50 for row in rows]
    assert p[0] <= p[1] <= p[2]
    assert rows[0].median_max_drawdown < rows[2].median_max_drawdown


def test_ruin_detected_when_a_period_wipes_the_book():
    r = np.full(400, 0.001)
    r[::50] = -0.4  # at 3x a -40% period is a -120% loss
    rows = sizing_stress(r, n_paths=500)
    assert rows[0].prob_ruin == 0.0
    assert rows[2].prob_ruin > 0.9
