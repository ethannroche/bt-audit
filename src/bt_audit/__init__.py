"""bt-audit: grade a trading backtest before trusting it."""

__version__ = "0.1.0"

from .lookahead import scan
from .report import run_audit, to_markdown

__all__ = ["run_audit", "to_markdown", "scan", "__version__"]
