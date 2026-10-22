# bt-audit: spec v0

Working name. Owner picks the public name before the repo is created.

## Problem

Hundreds of public trading-agent repos show a backtest and a headline Sharpe ratio. Almost none
answer the one question that matters: would this return survive real money? The usual failure
is not a missing stop-loss. It is one of five things:

1. **Selection bias.** The author tried many variants and published the best. A high Sharpe
   picked from N trials is expected even with zero edge.
2. **Short or fat-tailed track record.** The Sharpe estimate is too noisy to distinguish from
   zero.
3. **Costs.** Turnover times realistic fees and slippage eats the edge.
4. **Sizing.** Leverage and position sizing turn a positive-expectancy series into a path that
   hits ruin first.
5. **Lookahead.** The code uses information that was not available at decision time.

bt-audit grades a strategy on all five and prints a report card. It never produces signals,
never recommends a trade, and never says a strategy "will make money". It reports how much
evidence the backtest contains.

## Inputs

- `returns.csv` (required): a date column plus one column of per-period strategy returns
  (simple returns, net or gross). Optional extra columns = other trials the author tried.
- `--trials trials.csv` (optional): per-period returns of every variant tested (T rows x N
  columns). Enables PBO and the trial-adjusted Sharpe.
- `--n-trials N` (optional): number of variants tried when the matrix is not available.
- `--turnover x` (optional): average fraction of the book traded per period. Enables cost
  stress.
- `--periods-per-year` (default 252).
- `--repo PATH` (optional): source tree to scan for lookahead patterns.

## Statistics (all formulas cited in code)

| Check | Method | Source |
|---|---|---|
| Sharpe significance | Probabilistic Sharpe Ratio (PSR) against SR*=0, skew and kurtosis adjusted | Bailey and Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier" |
| Track-record length | Minimum Track Record Length (MinTRL) at 95% | same |
| Selection bias | Deflated Sharpe Ratio (DSR): PSR against the expected maximum Sharpe of N zero-edge trials | Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio" |
| Overfitting | Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation, S=16 blocks | Bailey, Borwein, Lopez de Prado, Zhu (2015/2017), "The Probability of Backtest Overfitting" |
| Costs | Sharpe at 0/5/10/20/50 bps per unit turnover; breakeven cost in bps | arithmetic |
| Sizing | Stationary block bootstrap of returns (5,000 paths, block = sqrt(T)); max-drawdown distribution and P(drawdown >= 50%) at leverage 1x/2x/3x | Politis and Romano (1994) |
| Lookahead | Static scan for known patterns (negative shifts, forward indexing, same-bar close fills, full-sample normalisation) | heuristics, labelled "possible", never "proven" |

## Verdict rules

Each check gets PASS / WARN / FAIL. Thresholds are stated in the report next to the result.

- PSR >= 0.95 PASS, 0.80 to 0.95 WARN, else FAIL.
- DSR (needs N) same thresholds as PSR.
- PBO <= 0.25 PASS, 0.25 to 0.50 WARN, > 0.50 FAIL.
- Breakeven cost >= 20 bps PASS, 5 to 20 WARN, < 5 FAIL.
- P(drawdown >= 50%) at 1x <= 5% PASS, 5 to 20% WARN, > 20% FAIL.
- Lookahead: any high-confidence hit = WARN (never FAIL, because it is static analysis).

Overall grade: the worst single check. A strategy is only as trustworthy as its weakest test.

## Non-goals

No data fetching, no live trading, no strategy generation, no "fixed" parameters. The auditor
measures; it does not optimise. Optimising against the audit would recreate the overfitting
the audit exists to catch.

## Acceptance

1. Pure-noise trials (N=50, T=1,000): PBO near 0.5 on average over seeds (tested: 0.35 to 0.65),
   and the cherry-picked winner gets an overall FAIL.
2. A planted true edge (per-period SR 0.15, T=1,000, one of N=50 trials): PBO under 0.1 on
   average over seeds.
3. Formulas behave correctly under simulated nulls (PSR size 5%, expected max Sharpe within 5% of brute force, Deflated Sharpe false-pass rate <= 5% at correlation 0, 0.5, 0.9).
4. Runs end to end on real data (125 years of S&P 500 timing variants) and real code (lookahead scan of public trading repos, with a labelled benchmark).
