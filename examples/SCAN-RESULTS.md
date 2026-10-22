# Lookahead scan of public trading repos (2026-10-22)

Shallow clones of each default branch, scanned with `bt_audit.lookahead.scan`. Every
high-confidence hit was then read by hand. "Verdict" is that reading, not the scanner's.

| Repo | Hits high / medium | Hand-checked verdict on the high hits |
|---|---|---|
| AI4Finance-Foundation/FinRL | 2 / 7 | **1 real leak, two effects** (re-checked on main c0018c2, 2026-10-22). `fundamental_stock_trading.py:306` backfills quarterly ratios keyed on Compustat `datadate`, the fiscal period END, so each quarter's EPS, book value and dividends appear from the day after the previous quarter end: about 3 months before the period closes and 4 to 5 before the 10-Q. The `bfill` also runs across the whole frame, so days after a ticker's last report take the next ticker's values. Both reproduced on a 10-line pandas example. The other high hit backfills the DJI benchmark while aligning backtest results; that distorts the reported comparison, not the agent's decisions. Most medium hits are `ffill().bfill()`, which leaks only the first value of each series. |
| freqtrade/freqtrade-strategies | 4 / 1 | **All 4 labelled leaky files flagged** (5 hits; Zeus has two) in the maintainers' own `lookahead_bias/` practice folder. Nothing flagged in the other 65 strategy files. |
| microsoft/qlib | 3 / 7 | **0 bugs, 2 hazards.** `BFillNan` is an operator that backfills by design; it leaks only if a user builds a feature with it, so it is worth knowing about rather than a defect. `utils/resam.py:225` takes the first valid value of a series, a utility with no time direction. Medium hits are model-fitting and reporting code inside a framework that fits on a train window. |
| TauricResearch/TradingAgents | 0 / 1 | `ffill().bfill()` on OHLCV: leaks only the first bar of a series. Minor. |
| virattt/ai-hedge-fund | 0 / 0 | Nothing flagged. |
| asavinov/intelligent-trading-bot | 0 / 0 | Nothing flagged at medium or high. |

What this says about the scanner: on strategy code (FinRL examples, freqtrade) a high hit was
usually real; on a framework (qlib) a high hit usually marks a tool that CAN leak, not a leak.
Treat the output as a list of places to read, ranked.
