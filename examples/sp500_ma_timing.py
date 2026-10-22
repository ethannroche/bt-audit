"""Real-data demo: 60 moving-average timing rules on 125 years of the S&P 500.

The classic trade: hold the index when its price is above its N-month moving average, sit in
cash when below. Try N = 2..61, keep the best, and publish it. This script builds every
variant's ACTIVE return (timing strategy minus buy-and-hold), so the audit asks the honest
question: does the timing add anything, after you account for having tried 60 versions?

Data: Shiller's monthly S&P 500 series via github.com/datasets/s-and-p-500 (data/data.csv).
No lookahead: the position for month t+1 is set from prices known at the end of month t.

    python examples/sp500_ma_timing.py path/to/data.csv out_dir/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build(csv: str) -> tuple[pd.DataFrame, float]:
    d = pd.read_csv(csv, parse_dates=["Date"]).set_index("Date")
    d = d[(d.index >= "1900-01-01") & (d["SP500"] > 0)]
    # the newest months carry 0.0 placeholders for dividends and rates until Shiller's data is
    # filled in; treating them as real zeros would understate both the index and the cash leg
    d = d[(d["Dividend"] > 0) & (d["Long Interest Rate"] > 0)]
    px = d["SP500"]
    total = px.pct_change() + (d["Dividend"] / 12.0) / px.shift(1)  # price + dividend yield
    cash = d["Long Interest Rate"] / 1200.0  # monthly cash proxy (long rate; flagged in README)
    out, switches = {}, []
    for n in range(2, 62):
        ma = px.rolling(n).mean()
        pos = (px > ma).astype(float).shift(1)  # decided at month-end t, held through t+1
        strat = pos * total + (1 - pos) * cash.shift(1)
        out[f"ma{n}"] = strat - total  # active return vs buy-and-hold
        switches.append(pos.diff().abs().mean())
    df = pd.DataFrame(out).dropna()
    return df, float(np.nanmean(switches))


if __name__ == "__main__":
    src, dst = sys.argv[1], Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    df, turnover = build(src)
    sr = df.mean() / df.std()
    best = sr.idxmax()
    df.to_csv(dst / "sp500_ma_trials.csv", index_label="date")
    print(f"rows={len(df)} variants={df.shape[1]} best={best} "
          f"best_active_sharpe_annual={sr[best] * np.sqrt(12):.2f} avg_turnover={turnover:.3f}")
