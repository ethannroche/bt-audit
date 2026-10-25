# bt-audit

Grades a trading backtest on one question: **would this return survive real money?**

Most public trading bots publish a backtest and a Sharpe ratio. Almost none correct for the things
that usually make that number fake. bt-audit runs seven checks for them and prints a report card with
the rule behind every grade.

| Check | What it catches | Method |
|---|---|---|
| Sharpe significance | a track record too short or too fat-tailed to tell from luck | Probabilistic Sharpe Ratio, Minimum Track Record Length |
| Selection bias | "I tried 60 versions and published the best one" | Deflated Sharpe Ratio, adjusted for how correlated the variants are |
| Selection bias, resampled | the same, without assuming normal returns or a correlation summary | Hansen's SPA test (stationary bootstrap), with White's Reality Check alongside |
| Overfitting | an in-sample winner that does not stay a winner | Probability of Backtest Overfitting (CSCV, 12,870 splits) |
| Costs | an edge that fees and slippage eat | Sharpe by cost, breakeven cost in bps |
| Sizing | leverage that turns a good average into ruin | stationary block bootstrap of drawdowns at 1x/2x/3x |
| Lookahead | code that reads the future | static scan of `.py` and `.ipynb` for known leak patterns |

It measures evidence. It does not generate strategies, tune parameters, fetch prices or trade,
and nothing it prints is investment advice.

## Install and run

```bash
pip install git+https://github.com/ethannroche/bt-audit   # not on PyPI yet
bt-audit returns.csv --trials all_variants.csv --turnover 0.3 --repo ./strategy-src -o report.md
```

`returns.csv`: a date column plus per-period returns. Extra numeric columns are treated as the
other variants you tried. Use `--equity` for an equity curve, `--active` for returns versus a
benchmark, `--periods-per-year 12` for monthly data, `--json` for machine-readable output. Exit code: 0 PASS, 1 WARN, 2 FAIL, so it
drops into CI.

## In CI (GitHub Action)

```yaml
- uses: ethannroche/bt-audit@main
  id: audit
  with:
    returns: results/returns.csv
    trials: results/all_variants.csv   # enables the Deflated Sharpe, PBO and SPA
    turnover: "0.3"
    fail-on: FAIL   # or WARN, or NONE to report only
```

The report card lands in the job summary; the job fails at the grade you choose, and
`steps.audit.outputs.grade-code` carries 0/1/2. Other inputs: `n-trials` (count of variants when
you have no trials CSV), `equity`, `active`, `column`, `periods-per-year`, `repo-path`. The
repo's own CI runs the action on 20 noise variants: the cherry-picked one passes the naive
Sharpe test (PSR 0.976) and fails the Deflated Sharpe (0.533) and
the SPA test (p 0.283).

## Real results

**1. "Hold the S&P 500 only when it is above its moving average."** The classic timing rule,
on Shiller's monthly data from 1900 to June 2023 (the newer months still carry zero
placeholders for dividends and rates, so they are left out), with every lookback from 2 to 61
months. The best variant (2 months) beats buy-and-hold with an active Sharpe of 0.20 and a
naive significance of 99.0%. After correcting for having tried 60 versions (mean correlation
0.84), the Deflated Sharpe is 0.910: **suggestive, not proven at 95%**. Its edge also needs costs
under 40 bps per unit traded.
Reproduce: `python examples/sp500_ma_timing.py data.csv out/`, report in
[`examples/sp500_ma_report.md`](examples/sp500_ma_report.md).

**2. A lookahead leak in a popular open-source trading-agent framework.** The scanner flagged
`processed_full.bfill(axis="rows")` in FinRL's fundamental stock-trading example
(`finrl/applications/stock_trading/fundamental_stock_trading.py`). Reading the code confirms it:
quarterly ratios (EPS, book value, dividends) are backfilled onto every earlier calendar day, so
the agent trades on each quarter's numbers before the quarter is reported. Six public repos scanned and hand-checked: [`examples/SCAN-RESULTS.md`](examples/SCAN-RESULTS.md).

**3. The lookahead scanner on a labelled set.** freqtrade-strategies keeps four strategies its
maintainers marked as leaking the future. The scanner flags all four at medium or high
confidence and flags nothing at that level in the other 65 strategy files.
`python benchmarks/freqtrade_lookahead.py` clones the repo (GPL-3.0, never copied here) and
reruns it. Honest caveat: the min-max rule was written after reading that set, so 4/4 is not a
held-out score.

## How the statistics were verified

Against simulation, not remembered examples: the PSR test rejects at 5% under a true null; the
closed-form expected maximum Sharpe matches brute force within 5%; PBO sits near 0.5 on pure
noise and near 0 when one variant carries a real edge; the mean-correlation estimate recovers a
planted 0.5. An independent review checked every formula against the papers; its one real finding (an anti-conservative Deflated Sharpe on correlated variants) is fixed and pinned by a regression test.

The SPA test is checked the same way, with 300 bootstrap resamples per p-value in tests (the
report uses 2,000). Its PASS cut is p <= 0.025, not 0.05, because the test runs hot in finite
samples. Simulated on 20 zero-edge variants at T = 1,000 (600 runs each), the share graded PASS is:

| Returns | p <= 0.05 | p <= 0.025 (the PASS cut) |
|---|---|---|
| independent | 7.8% | 3.5% |
| pairwise correlation 0.5 | 7.7% | 3.3% |
| fat-tailed (t, 4 df) | 8.0% | 4.8% |
| AR(1) 0.2 | 9.0% | 5.2% |
| AR(1) 0.5 | 13.5% | 7.3% |
| independent, T = 250 | 8.3% | 5.3% |
| AR(1) 0.2, T = 250 | 12.3% | 7.3% |

One variant with a per-period Sharpe of 0.12 among 10 (T = 2,000) still passes in 60 of 60 runs.
A weaker edge is mostly missed: per-period Sharpe 0.06 (about 1.0 annualised on daily data) among 20
at T = 1,000 passes in 12.5% of 200 runs at the 0.025 cut (24.5% at 0.05). The stricter cut buys a
lower false-pass rate with that power. With one modest winner among 29 clearly losing
variants, SPA_c rejects far more often than White's Reality Check, which is the failure SPA was
designed to fix. Independent review also found that twenty variants that all lose money got
p = 0.0 and a PASS (a strict inequality in the p-value); ties now count against rejection and a
regression test pins it. Removing the studentization, the recentring, the SPA_c trimming or the
tie rule each makes a test fail. `pytest` runs all of it (49 tests, about 70 seconds), and CI
also runs the GitHub Action end to end.

## Known limits

- The Deflated Sharpe takes V[SR] at its zero-edge value, (1 - rho)/(T - 1), with rho the mean
  pairwise correlation of the variants, instead of the trials' observed Sharpe dispersion. The
  observed dispersion lets one genuine edge among the trials raise the bar for that same
  strategy. Simulated false-pass rate on zero-edge trials stays at or under the nominal 5% at
  correlations 0, 0.5 and 0.9 (tested). This deviates from the paper and the report says so.
- Lookahead findings are static pattern matches: "look here", never "proven". Test files and
  test folders are skipped.
- SPA on raw returns tests whether any variant beats a zero return (cash), which any long-only
  drift does; pass `--active` returns to test against a benchmark. The report says which. A low
  p-value does not say the audited column is the variant that does. Variants missing more than
  10% of periods are left out, and the report says how many. Its mean block length is the cube root of the sample, a fixed
  rule; the square-root rule used for sizing made it reject about 10% of the time at a 5% level on
  independent noise.
- The bootstrap compounds simple returns and ignores financing, so leveraged rows are optimistic.
- Power: a real per-period Sharpe of 0.12 (about 1.9 annualised on daily data), hidden among
  10 variants, clears the deflated test in 40 of 40 simulated runs at 2,000 periods.

## Related tools

- **overfit-gauntlet** (MIT, July 2026): PSR, DSR, permutation and bootstrap tests, walk-forward
  and purged k-fold, cost stress, and an AST lookahead linter. The closest neighbour. bt-audit
  adds PBO via CSCV, a correlation-aware Deflated Sharpe with a tested false-pass rate, bootstrap
  drawdown and ruin by leverage, and published evidence (a labelled benchmark and hand-checked
  scans of public repos). Compared head to head in [`examples/VS-OVERFIT-GAUNTLET.md`](examples/VS-OVERFIT-GAUNTLET.md): on cherry-picked noise and a planted edge the two tie at their top grade (neither validates noise, both validate the edge); on freqtrade's four labelled leaky strategies bt-audit flags 4 and overfit-gauntlet 0, with the caveat that bt-audit's rule was written after seeing that set. Rerun with `bash benchmarks/run_vs_overfit_gauntlet.sh`.
- **backtest-reality-check** (MIT, MCP server): deflated Sharpe, noise Sharpe, track-record
  length and fill realism, for use inside an AI assistant. No overfitting or lookahead checks.
- **pypbo**: a Python library for PBO and related statistics.

## References

- Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk* 15(2).
- Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5).
- Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance* 20(4).
- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of Business & Economic Statistics* 23(4).
- Politis, D. N. and Romano, J. P. (1994). The Stationary Bootstrap. *JASA* 89(428).
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica* 68(5).

MIT licensed.
