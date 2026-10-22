import numpy as np
import pandas as pd

from bt_audit.cli import main


def _csv(tmp_path, arr, name):
    df = pd.DataFrame(arr, columns=[f"v{i}" for i in range(arr.shape[1])])
    df.insert(0, "date", pd.date_range("2020-01-01", periods=len(df), freq="B").astype(str))
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_cherry_picked_noise_fails(tmp_path, capsys):
    x = np.random.default_rng(0).normal(0, 0.01, (1000, 40))
    best = int((x.mean(0) / x.std(0)).argmax())
    p = _csv(tmp_path, x, "trials.csv")
    code = main([str(p), "--column", f"v{best}", "--turnover", "0.5"])
    out = capsys.readouterr().out
    assert code == 2
    assert "Overall: FAIL" in out
    assert "Overfitting (PBO, CSCV) | FAIL" in out or "Selection bias (Deflated Sharpe) | FAIL" in out


def test_real_edge_with_low_costs_passes_stat_checks(tmp_path, capsys):
    rng = np.random.default_rng(1)
    # 6,000 periods: the deflated test is conservative, and at 2,000 periods this edge clears
    # it only ~85% of the time (measured over 20 seeds). That is statistical power, not a bug.
    x = rng.normal(0, 0.01, (6000, 10))
    x[:, 0] += 0.0012
    p = _csv(tmp_path, x, "trials.csv")
    main([str(p), "--column", "v0", "--turnover", "0.05"])
    out = capsys.readouterr().out
    assert "Sharpe significance (PSR) | PASS" in out
    assert "Selection bias (Deflated Sharpe) | PASS" in out
    assert "Overfitting (PBO, CSCV) | PASS" in out
    assert "Costs | PASS" in out


def test_equity_curve_input(tmp_path, capsys):
    rng = np.random.default_rng(2)
    eq = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 800))
    p = _csv(tmp_path, eq[:, None], "equity.csv")
    main([str(p), "--equity"])
    assert "Headline Sharpe" in capsys.readouterr().out


def test_json_output(tmp_path, capsys):
    import json

    x = np.random.default_rng(3).normal(0.0005, 0.01, (600, 3))
    p = _csv(tmp_path, x, "t.csv")
    main([str(p), "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["grade"] in {"PASS", "WARN", "FAIL"}
    assert {c["name"] for c in d["checks"]} >= {"Sharpe significance (PSR)", "Overfitting (PBO, CSCV)"}
