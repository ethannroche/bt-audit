"""Static scan for lookahead (future-information) patterns in Python trading code.

This is heuristic. A hit means "look here", never "proven leak": a negative shift that builds a
training label is legitimate, the same shift inside a signal is not. Findings are labelled with a
confidence and a one-line reason so a reader can decide in seconds.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__", "site-packages", "build", "dist",
             "tests", "test", "unit_tests"}


def _is_test_file(p: Path) -> bool:
    return p.name.startswith("test_") or p.stem.endswith("_test")


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern
    confidence: str  # high | medium | low
    reason: str


RULES: tuple[Rule, ...] = (
    Rule(
        "bfill-after-ffill",
        re.compile(
            r"ffill\(\)\s*\.bfill\("
            r"|method\s*=\s*['\"](?:ffill|pad)['\"]\s*\)\s*\.fillna\([^)]*method\s*=\s*['\"](?:bfill|backfill)"
        ),
        "medium",
        "after a forward fill, the backfill only reaches leading gaps: each series' start borrows its first future value",
    ),
    Rule(
        "bfill",
        re.compile(
            r"(?<!ffill\(\))\.bfill\("
            r"|(?<!ffill['\"]\))(?<!pad['\"]\))\.fillna\([^)]*method\s*=\s*['\"](?:bfill|backfill)"
        ),
        "high",
        "backfill copies a later value into an earlier row",
    ),
    Rule(
        "full-sample-minmax",
        re.compile(r"\b([\w\[\]'\"]+)\.max\(\)\s*-\s*\1\.min\(\)"),
        "high",
        "min-max scaling with whole-series max and min uses the future range at every point",
    ),
    Rule(
        "centered-window",
        re.compile(r"rolling\([^)]*center\s*=\s*True"),
        "high",
        "a centred rolling window averages in future rows",
    ),
    Rule(
        "shuffled-split",
        re.compile(r"train_test_split\((?![^)]*shuffle\s*=\s*False)"),
        "high",
        "train_test_split shuffles by default, mixing future rows into training",
    ),
    Rule(
        "kfold-on-time-series",
        re.compile(r"\b(?:KFold|StratifiedKFold|cross_val_score)\("),
        "medium",
        "non-temporal cross-validation trains on the future; time series need walk-forward splits",
    ),
    Rule(
        "negative-shift",
        re.compile(r"\.shift\(\s*-\s*\d+"),
        "medium",
        "negative shift pulls future values back; fine for a label, a leak in a feature or signal",
    ),
    Rule(
        "full-sample-fit",
        re.compile(r"\.fit_transform\(|\.fit\(\s*(?:df|data|X)\s*\)"),
        "medium",
        "a scaler or model fitted on the full sample sees test-period statistics",
    ),
    Rule(
        "full-sample-zscore",
        re.compile(r"\(\s*[\w\[\]'\"]+\s*-\s*[\w\[\]'\"]+\.mean\(\)\s*\)\s*/\s*[\w\[\]'\"]+\.std\(\)"),
        "medium",
        "a z-score with whole-series mean and std uses future data at every point",
    ),
    Rule(
        "forward-index",
        re.compile(r"\[\s*(?:i|t|idx|index)\s*\+\s*1\s*\]"),
        "low",
        "indexing i+1 inside a loop may read the next bar before acting on this one",
    ),
)


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    rule: str
    confidence: str
    reason: str
    code: str


def _iter_sources(root: Path):
    for p in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts) or _is_test_file(p):
            continue
        if p.suffix == ".py" and p.is_file():
            try:
                yield p, p.read_text(errors="replace").splitlines()
            except OSError:
                continue
        elif p.suffix == ".ipynb" and p.is_file():
            try:
                nb = json.loads(p.read_text(errors="replace"))
            except (OSError, json.JSONDecodeError):
                continue
            lines: list[str] = []
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    src = cell.get("source", [])
                    lines.extend((src if isinstance(src, list) else src.splitlines(True)))
            yield p, [ln.rstrip("\n") for ln in lines]


# A groupby on the date axis means the statistic is taken ACROSS assets at one instant
# (cross-sectional), which uses no future data.
_CROSS_SECTIONAL = re.compile(r"groupby\([^)]*['\"](?:date|datetime|time|timestamp|day)['\"]|level\s*=\s*['\"]?(?:date|datetime|0)")
_NORMALISING_RULES = {"full-sample-zscore", "full-sample-minmax", "full-sample-fit"}
# Reporting, plotting and analysis code describes results after the fact; a leak there does not
# reach a trading decision. Hits are kept but drop one confidence level.
_REPORTING_PATH = re.compile(r"(?:^|/)(?:report|reports|plot|plots|plotting|viz|visuali[sz]ation|analysis\w*)(?:/|\.py$|_)", re.I)
_DOWN = {"high": "medium", "medium": "low", "low": "low"}


def scan(root: str | Path) -> list[Finding]:
    root = Path(root)
    out: list[Finding] = []
    for path, lines in _iter_sources(root):
        for no, raw in enumerate(lines, 1):
            code = raw.split("#", 1)[0]
            if not code.strip():
                continue
            rel = str(path.relative_to(root))
            for rule in RULES:
                if rule.pattern.search(code):
                    if rule.id in _NORMALISING_RULES and _CROSS_SECTIONAL.search(code):
                        continue
                    conf = _DOWN[rule.confidence] if _REPORTING_PATH.search(rel) else rule.confidence
                    out.append(
                        Finding(
                            file=rel,
                            line=no,
                            rule=rule.id,
                            confidence=conf,
                            reason=rule.reason,
                            code=raw.strip()[:160],
                        )
                    )
    return out
