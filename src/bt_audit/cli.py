"""bt-audit command line.

    bt-audit returns.csv [--column NAME] [--equity] [--trials trials.csv | --n-trials N]
             [--turnover X] [--periods-per-year 252] [--repo PATH] [-o report.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .lookahead import scan
import json

from .report import run_audit, to_dict, to_markdown


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.apply(pd.to_numeric, errors="coerce")
    return num.loc[:, num.notna().mean() > 0.9]


def load(path: str, equity: bool) -> pd.DataFrame:
    df = _numeric(pd.read_csv(path))
    if df.empty:
        raise SystemExit(f"{path}: no numeric columns found")
    if equity:
        df = df.pct_change().iloc[1:]
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bt-audit", description="Grade a backtest before trusting it.")
    ap.add_argument("returns", help="CSV of per-period returns (or an equity curve with --equity)")
    ap.add_argument("--column", help="strategy column (default: first numeric column)")
    ap.add_argument("--equity", action="store_true", help="input columns are equity/price levels, not returns")
    ap.add_argument("--trials", help="CSV of every variant's returns (T x N); enables PBO")
    ap.add_argument("--n-trials", type=int, help="number of variants tried, if no trials CSV")
    ap.add_argument("--turnover", type=float, help="average fraction of the book traded per period")
    ap.add_argument("--periods-per-year", type=int, default=252)
    ap.add_argument("--active", action="store_true", help="returns are active (strategy minus benchmark); skips sizing")
    ap.add_argument("--repo", help="source tree to scan for lookahead patterns")
    ap.add_argument("-o", "--output", help="write the markdown report here")
    ap.add_argument("--json", action="store_true", help="print the report as JSON instead of markdown")
    a = ap.parse_args(argv)

    df = load(a.returns, a.equity)
    col = a.column or df.columns[0]
    if col not in df.columns:
        raise SystemExit(f"column {col!r} not found; numeric columns: {list(df.columns)}")
    returns = df[col].to_numpy(dtype=float)

    trials = None
    if a.trials:
        trials = load(a.trials, a.equity).to_numpy(dtype=float)
    elif df.shape[1] > 1:
        trials = df.to_numpy(dtype=float)

    findings = scan(Path(a.repo)) if a.repo else None
    audit = run_audit(returns, trials, a.n_trials, a.turnover, a.periods_per_year, findings, a.active)
    md = to_markdown(audit, title=f"Backtest audit: {Path(a.returns).name} [{col}]")
    if a.output:
        Path(a.output).write_text(md)
    sys.stdout.write(json.dumps(to_dict(audit), indent=2) + "\n" if a.json else md)
    return {"PASS": 0, "WARN": 1, "FAIL": 2}[audit.grade]


if __name__ == "__main__":
    raise SystemExit(main())
