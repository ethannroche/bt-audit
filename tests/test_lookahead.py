from bt_audit.lookahead import scan


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_flags_known_leaks(tmp_path):
    _write(
        tmp_path,
        "strat.py",
        "\n".join(
            [
                "df['sig'] = df['close'].rolling(20, center=True).mean()",
                "df = df.bfill()",
                "X_tr, X_te = train_test_split(X, y, test_size=0.2)",
                "df['fwd'] = df['close'].shift(-1)",
                "z = (df['x'] - df['x'].mean()) / df['x'].std()",
                "for i in range(n): buy = price[i + 1] > price[i]",
            ]
        ),
    )
    rules = {f.rule for f in scan(tmp_path)}
    assert {"centered-window", "bfill", "shuffled-split", "negative-shift", "full-sample-zscore", "forward-index"} <= rules


def test_safe_split_and_comments_are_not_flagged(tmp_path):
    _write(
        tmp_path,
        "ok.py",
        "\n".join(
            [
                "X_tr, X_te = train_test_split(X, y, shuffle=False)",
                "# df = df.bfill()   (commented out)",
                "df['ma'] = df['close'].rolling(20).mean()",
                "df['prev'] = df['close'].shift(1)",
            ]
        ),
    )
    assert scan(tmp_path) == []


def test_reads_notebooks_and_skips_vendor_dirs(tmp_path):
    import json

    nb = {"cells": [{"cell_type": "code", "source": ["df = df.bfill()\n"]}]}
    (tmp_path / "nb.ipynb").write_text(json.dumps(nb))
    (tmp_path / ".venv").mkdir()
    _write(tmp_path / ".venv", "lib.py", "df = df.bfill()")
    hits = scan(tmp_path)
    assert [h.file for h in hits] == ["nb.ipynb"]


def test_minmax_and_fill_rules(tmp_path):
    _write(
        tmp_path,
        "s.py",
        "\n".join(
            [
                "df = (df-df.min())/(df.max()-df.min())",
                "w = (x - x.rolling(20).min())/(x.rolling(20).max()-x.rolling(20).min())",
                "a = a.ffill().bfill()",
                "b = b.shift(1).fillna(method='bfill')",
            ]
        ),
    )
    got = [(f.line, f.rule, f.confidence) for f in scan(tmp_path)]
    assert got == [
        (1, "full-sample-minmax", "high"),
        (3, "bfill-after-ffill", "medium"),
        (4, "bfill", "high"),
    ]


def test_skips_test_files(tmp_path):
    _write(tmp_path, "test_x.py", "df = df.bfill()")
    (tmp_path / "tests").mkdir()
    _write(tmp_path / "tests", "helpers.py", "df = df.bfill()")
    assert scan(tmp_path) == []
