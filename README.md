# bt-audit

Grades a trading backtest on one question: **would this return survive real money?**

Most public trading bots publish a backtest and a Sharpe ratio. Almost none correct for the five
things that usually make that number fake. bt-audit checks all five and prints a report card
with the rule behind every grade.

| Check | What it catches | Method |
|---|---|---|
| Sharpe significance | a track record too short or too fat-tailed to tell from luck | Probabilistic Sharpe Ratio, Minimum Track Record Length |
| Selection bias | "I tried 60 versions and published the best one" | Deflated Sharpe Ratio, adjusted for how correlated the variants are |
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

## Real results

**1. "Hold the S&P 500 only when it is above its moving average."** The classic timing rule,
on 125 years of Shiller's monthly data (1900 to 2025), with every lookback from 2 to 61 months.
The best variant (2 months) beats buy-and-hold with an active Sharpe of 0.18 and a naive
significance of 98.4%. After correcting for having tried 60 versions (mean correlation 0.84),
the Deflated Sharpe is 0.878: **suggestive, not proven at 95%**. Its edge also needs costs under
36 bps per unit traded.
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
noise and near 0 when one variant carries a real edge; the mean-correlation estimate recovers a planted 0.5.
An independent review checked every formula against the papers; its one real finding (an anti-conservative Deflated Sharpe on correlated variants) is fixed and pinned by a regression test. `pytest` runs all of it (33 tests).

## Known limits

- The Deflated Sharpe takes V[SR] at its zero-edge value, (1 - rho)/(T - 1), with rho the mean
  pairwise correlation of the variants, instead of the trials' observed Sharpe dispersion. The
  observed dispersion lets one genuine edge among the trials raise the bar for that same
  strategy. Simulated false-pass rate on zero-edge trials stays at or under the nominal 5% at
  correlations 0, 0.5 and 0.9 (tested). This deviates from the paper and the report says so.
- Lookahead findings are static pattern matches: "look here", never "proven". Test files and
  test folders are skipped.
- The bootstrap compounds simple returns and ignores financing, so leveraged rows are optimistic.
- The deflated test is conservative: a real per-period Sharpe of 0.12 clears it about 85% of the
  time at 2,000 periods and every time at 6,000.

## Related tools

- **overfit-gauntlet** (MIT, July 2026): PSR, DSR, permutation and bootstrap tests, walk-forward
  and purged k-fold, cost stress, and an AST lookahead linter. The closest neighbour. bt-audit
  adds PBO via CSCV, a correlation-aware Deflated Sharpe with a tested false-pass rate, bootstrap
  drawdown and ruin by leverage, and published evidence (a labelled benchmark and hand-checked
  scans of public repos). Not compared head to head yet.
- **backtest-reality-check** (MIT, MCP server): deflated Sharpe, noise Sharpe, track-record
  length and fill realism, for use inside an AI assistant. No overfitting or lookahead checks.
- **pypbo**: a Python library for PBO and related statistics.

## References

- Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk* 15(2).
- Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5).
- Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance* 20(4).
- Politis, D. N. and Romano, J. P. (1994). The Stationary Bootstrap. *JASA* 89(428).

MIT licensed.
