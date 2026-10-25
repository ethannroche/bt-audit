#!/bin/bash
# Run from your laptop terminal, in the bt-audit folder:  bash benchmarks/run_vs_overfit_gauntlet.sh
# Makes a throwaway venv, installs this checkout and overfit-gauntlet 0.1.0 (MIT, numpy + scipy),
# runs the comparison (about 10 minutes) and writes examples/VS-OVERFIT-GAUNTLET.md.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-vs
.venv-vs/bin/pip install -q -e . "overfit-gauntlet==0.1.0"
PATH="$PWD/.venv-vs/bin:$PATH" .venv-vs/bin/python benchmarks/vs_overfit_gauntlet.py "$@"
echo "DONE. Commit examples/VS-OVERFIT-GAUNTLET.md, or tell Claude 'gauntlet run done'."
