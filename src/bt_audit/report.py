"""Runs every check and renders the report card. Verdict thresholds live here and are printed
next to each result, so a reader never has to trust a grade they cannot see the rule for."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import pbo as pbo_mod
from . import stats, stress
from .lookahead import Finding

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
_ORDER = {PASS: 0, SKIP: 0, WARN: 1, FAIL: 2}


@dataclass
class Check:
    name: str
    verdict: str
    result: str
    rule: str
    detail: list[str] = field(default_factory=list)


@dataclass
class Audit:
    checks: list[Check]
    periods: int
    periods_per_year: int
    sharpe_annual: float
    lookahead: list[Finding]

    @property
    def grade(self) -> str:
        worst = max((_ORDER[c.verdict] for c in self.checks), default=0)
        return {0: PASS, 1: WARN, 2: FAIL}[worst]


def _band(x: float, pass_at: float, warn_at: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return PASS if x >= pass_at else WARN if x >= warn_at else FAIL
    return PASS if x <= pass_at else WARN if x <= warn_at else FAIL


def run_audit(
    returns: np.ndarray,
    trials: np.ndarray | None = None,
    n_trials: int | None = None,
    turnover: float | None = None,
    periods_per_year: int = 252,
    lookahead: list[Finding] | None = None,
    active: bool = False,
) -> Audit:
    m = stats.moments(returns)
    sr_ann = stats.annualise(m.sharpe, periods_per_year)
    checks: list[Check] = []

    p = stats.psr(m)
    mintrl = stats.min_track_record_length(m)
    checks.append(
        Check(
            "Sharpe significance (PSR)",
            _band(p, 0.95, 0.80),
            f"{p:.3f} probability the true Sharpe is above zero",
            "PASS >= 0.95, WARN >= 0.80",
            [
                f"Observed Sharpe {sr_ann:.2f} annualised over {m.n} periods "
                f"(skew {m.skew:.2f}, kurtosis {m.kurtosis:.2f}).",
                "Track record needed for 95% confidence: "
                + ("never, the observed Sharpe is not positive" if math.isinf(mintrl)
                   else f"{mintrl:,.0f} periods ({mintrl / periods_per_year:.1f} years); you have {m.n:,}."),
            ],
        )
    )

    n_raw = n_trials if n_trials else (trials.shape[1] if trials is not None else None)
    if n_raw and n_raw >= 2:
        # The paper's V[SR] is the cross-sectional variance of the trials' Sharpe estimates. We use
        # its value UNDER THE ZERO-EDGE NULL instead, (1 - rho)/(T - 1), with rho the trials' mean
        # pairwise correlation: the shared part of correlated trials moves them together and does
        # not raise the maximum. Using the observed dispersion would let one genuine edge among the
        # trials inflate the hurdle for the very strategy that has the edge.
        rho = stats.mean_pairwise_correlation(trials) if (trials is not None and not n_trials) else 0.0
        sr_var = (1.0 - rho) / (m.n - 1)
        d = stats.dsr(m, n_raw, sr_var)
        hurdle = stats.expected_max_sharpe(n_raw, sr_var)
        detail = [
            f"The best of {n_raw} zero-edge strategies like these would show a Sharpe of about "
            f"{stats.annualise(hurdle, periods_per_year):.2f} annualised by luck alone.",
            (f"Mean pairwise correlation of the variants: {rho:.2f} (shared moves do not raise the luck bar)."
             if trials is not None and not n_trials else
             "Variants assumed uncorrelated (only a count was given), which is the strictest reading."),
            "Method: Bailey and Lopez de Prado (2014) with V[SR] taken at its zero-edge value "
            "(1 - rho)/(T - 1) rather than the trials' observed dispersion.",
        ]
        checks.append(
            Check(
                "Selection bias (Deflated Sharpe)",
                _band(d, 0.95, 0.80),
                f"{d:.3f} after accounting for {n_raw} variants tried",
                "PASS >= 0.95, WARN >= 0.80",
                detail,
            )
        )
    else:
        checks.append(Check("Selection bias (Deflated Sharpe)", SKIP, "not run",
                            "needs --n-trials or a trials matrix",
                            ["Without the number of variants tried, the headline Sharpe cannot be "
                             "corrected for cherry-picking. Ask the author how many they tested."]))

    if trials is not None and trials.shape[1] >= 2:
        try:
            r = pbo_mod.pbo(trials)
            checks.append(
                Check(
                    "Overfitting (PBO, CSCV)",
                    _band(r.pbo, 0.25, 0.50, higher_is_better=False),
                    f"{r.pbo:.0%} of {r.n_splits:,} splits picked a below-median strategy out of sample",
                    "PASS <= 25%, WARN <= 50%",
                    ["PBO measures whether the selection process is stable (does the in-sample "
                     "winner stay near the top out of sample), not whether there is an edge. "
                     "Near-identical variants score low here by construction; read it with the "
                     "Deflated Sharpe above.",
                     f"The in-sample winner lost money out of sample in {r.prob_oos_loss:.0%} of splits.",
                     *([f"The oldest {r.rows_dropped} rows were trimmed so all 16 blocks are equal."] if r.rows_dropped else []),
                     f"Median Sharpe of the winner: in-sample {np.median(r.is_sharpe_best) * math.sqrt(periods_per_year):.2f}, "
                     f"out-of-sample {np.median(r.oos_sharpe_best) * math.sqrt(periods_per_year):.2f} (annualised)."],
                )
            )
        except ValueError as e:
            checks.append(Check("Overfitting (PBO, CSCV)", SKIP, "not run", "needs >= 32 rows", [str(e)]))
    else:
        checks.append(Check("Overfitting (PBO, CSCV)", SKIP, "not run", "needs a trials matrix",
                            ["Export the returns of every variant tested to measure overfitting directly."]))

    if turnover:
        c = stress.cost_stress(returns, turnover, periods_per_year)
        grid = ", ".join(f"{b:g} bps: {s:.2f}" for b, s in zip(c.bps_grid, c.sharpe_annual))
        checks.append(
            Check(
                "Costs",
                _band(c.breakeven_bps, 20, 5),
                f"edge disappears at {c.breakeven_bps:.1f} bps per unit turnover",
                "PASS >= 20 bps, WARN >= 5 bps",
                [f"Annualised Sharpe by cost ({turnover:g} turnover per period): {grid}.",
                 "Returns are treated as gross of costs."],
            )
        )
    else:
        checks.append(Check("Costs", SKIP, "not run", "needs --turnover",
                            ["Pass the average fraction of the book traded per period."]))

    if active:
        checks.append(Check("Sizing (bootstrap drawdowns)", SKIP, "not run",
                            "needs the strategy's own returns, not active returns",
                            ["Active returns (strategy minus benchmark) are not a wealth path, so "
                             "compounding them gives meaningless drawdowns. Audit the strategy's own "
                             "returns separately for sizing."]))
    else:
        rows = stress.sizing_stress(returns)
        one = rows[0]
        checks.append(
            Check(
                "Sizing (bootstrap drawdowns)",
                _band(one.prob_drawdown_50, 0.05, 0.20, higher_is_better=False),
                f"{one.prob_drawdown_50:.1%} chance of a 50% drawdown at 1x",
                "PASS <= 5%, WARN <= 20%",
                [f"{r.leverage:g}x: median max drawdown {r.median_max_drawdown:.0%}, "
                 f"worst 5% {r.p95_max_drawdown:.0%}, P(-50%) {r.prob_drawdown_50:.1%}, "
                 f"P(ruin) {r.prob_ruin:.1%}" for r in rows]
                + [f"Stationary block bootstrap, 5,000 paths, mean block length {math.sqrt(m.n):.0f} periods "
                   "(square root of the sample; a fixed rule, not tuned).",
                   "Returns are compounded as simple returns and leverage multiplies the whole return; "
                   "financing costs are ignored, so leveraged rows are optimistic."],
            )
        )

    if lookahead is not None:
        high = [f for f in lookahead if f.confidence == "high"]
        checks.append(
            Check(
                "Lookahead (static scan)",
                WARN if high else PASS,
                f"{len(high)} high, {sum(f.confidence == 'medium' for f in lookahead)} medium, "
                f"{sum(f.confidence == 'low' for f in lookahead)} low confidence hits",
                "WARN on any high-confidence hit (static analysis never FAILs)",
                [f"[{f.confidence}] {f.file}:{f.line} {f.rule}: {f.reason}. `{f.code}`" for f in lookahead[:25]]
                + ([f"... and {len(lookahead) - 25} more"] if len(lookahead) > 25 else []),
            )
        )

    return Audit(checks, m.n, periods_per_year, sr_ann, lookahead or [])


def to_markdown(a: Audit, title: str = "Backtest audit") -> str:
    out = [f"# {title}", "", f"**Overall: {a.grade}** (the weakest check sets the grade)", "",
           f"Headline Sharpe {a.sharpe_annual:.2f} annualised, {a.periods:,} periods.", "",
           "| Check | Verdict | Result | Rule |", "|---|---|---|---|"]
    for c in a.checks:
        out.append(f"| {c.name} | {c.verdict} | {c.result} | {c.rule} |")
    for c in a.checks:
        if c.detail:
            out += ["", f"## {c.name}", ""] + [f"- {d}" for d in c.detail]
    out += ["", "---", "",
            "This report measures how much evidence a backtest contains. It is not investment advice "
            "and says nothing about future returns."]
    return "\n".join(out) + "\n"


def to_dict(a: Audit) -> dict:
    """Machine-readable report (for CI and dashboards)."""
    return {
        "grade": a.grade,
        "sharpe_annual": a.sharpe_annual,
        "periods": a.periods,
        "periods_per_year": a.periods_per_year,
        "checks": [
            {"name": c.name, "verdict": c.verdict, "result": c.result, "rule": c.rule, "detail": c.detail}
            for c in a.checks
        ],
        "lookahead": [
            {"file": f.file, "line": f.line, "rule": f.rule, "confidence": f.confidence, "reason": f.reason}
            for f in a.lookahead
        ],
    }
