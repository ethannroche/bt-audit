# Backtest audit: sp500_ma_trials.csv [ma2]

**Overall: WARN** (the weakest check sets the grade)

Headline Sharpe 0.18 annualised, 1,520 periods.

| Check | Verdict | Result | Rule |
|---|---|---|---|
| Sharpe significance (PSR) | PASS | 0.984 probability the true Sharpe is above zero | PASS >= 0.95, WARN >= 0.80 |
| Selection bias (Deflated Sharpe) | WARN | 0.878 after accounting for 60 variants tried | PASS >= 0.95, WARN >= 0.80 |
| Overfitting (PBO, CSCV) | PASS | 0% of 12,870 splits picked a below-median strategy out of sample | PASS <= 25%, WARN <= 50% |
| Costs | PASS | edge disappears at 36.0 bps per unit turnover | PASS >= 20 bps, WARN >= 5 bps |
| Sizing (bootstrap drawdowns) | SKIP | not run | needs the strategy's own returns, not active returns |

## Sharpe significance (PSR)

- Observed Sharpe 0.18 annualised over 1520 periods (skew 1.87, kurtosis 17.57).
- Track record needed for 95% confidence: 901 periods (75.1 years); you have 1,520.

## Selection bias (Deflated Sharpe)

- The best of 60 zero-edge strategies like these would show a Sharpe of about 0.08 annualised by luck alone.
- Mean pairwise correlation of the variants: 0.84 (shared moves do not raise the luck bar).
- Method: Bailey and Lopez de Prado (2014) with V[SR] taken at its zero-edge value (1 - rho)/(T - 1) rather than the trials' observed dispersion.

## Overfitting (PBO, CSCV)

- PBO measures whether the selection process is stable (does the in-sample winner stay near the top out of sample), not whether there is an edge. Near-identical variants score low here by construction; read it with the Deflated Sharpe above.
- The in-sample winner lost money out of sample in 2% of splits.
- Median Sharpe of the winner: in-sample 0.20, out-of-sample 0.13 (annualised).

## Costs

- Annualised Sharpe by cost (0.4 turnover per period): 0 bps: 0.18, 5 bps: 0.16, 10 bps: 0.13, 20 bps: 0.08, 50 bps: -0.07.
- Returns are treated as gross of costs.

## Sizing (bootstrap drawdowns)

- Active returns (strategy minus benchmark) are not a wealth path, so compounding them gives meaningless drawdowns. Audit the strategy's own returns separately for sizing.

---

This report measures how much evidence a backtest contains. It is not investment advice and says nothing about future returns.
