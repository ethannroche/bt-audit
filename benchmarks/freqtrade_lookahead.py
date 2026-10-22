"""Lookahead-scanner benchmark against a labelled set.

freqtrade-strategies keeps a folder of strategies its maintainers marked as containing lookahead
bias, with the mistake explained in user_data/strategies/lookahead_bias/readme.md. That makes a
small ground-truth set: every file in it has a known leak.

The repo is GPL-3.0, so nothing from it is copied here. This script clones it at run time and
reports which labelled files the scanner catches.

    python benchmarks/freqtrade_lookahead.py [path/to/freqtrade-strategies]
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from bt_audit.lookahead import scan

URL = "https://github.com/freqtrade/freqtrade-strategies"
LABELLED = Path("user_data/strategies/lookahead_bias")


def main() -> int:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = Path(tempfile.mkdtemp()) / "freqtrade-strategies"
        subprocess.run(["git", "clone", "-q", "--depth", "1", URL, str(root)], check=True)
    target = root / LABELLED
    files = sorted(p.name for p in target.glob("*.py"))
    found = {f.file: f for f in scan(target) if f.confidence in ("high", "medium")}
    caught = 0
    for name in files:
        hit = found.get(name)
        caught += hit is not None
        print(f"{'CAUGHT' if hit else 'MISSED'}  {name:20s} {hit.rule + ' (line ' + str(hit.line) + ')' if hit else ''}")
    print(f"\nrecall: {caught}/{len(files)} labelled leaky strategies flagged at medium or high confidence")
    return 0 if caught == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
