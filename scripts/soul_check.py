#!/usr/bin/env python3
"""soul_check.py — the Kairos Constitution enforcer.

Kairos fuses two systems with *deliberately different* souls, so the
constitution is **scoped**:

* **System 1 — the engine** (``src/kairos/perception``, ``src/kairos/bridge``,
  the C++ ring): "Don't predict, understand." It refuses classic price-lagged
  technical analysis, copy-by-value on the hot path, REST in the execution
  path, and supervised regime labels.
* **System 2 — the reasoning firm** (``src/kairos/reasoning``): a multi-agent
  LLM that *may* reason about RSI/MACD/etc. as fallible evidence among many.
  It is intentionally EXEMPT from the no-classic-TA rule — that is its design.

RULES ENFORCED (engine scope unless noted)
------------------------------------------
* Rule 1  — No memcpy/memmove on the LOB hot path (C/C++); waive with
            ``soul:allow-memcpy``.
* Rule 2  — No classic / price-lagged technical analysis vocabulary.
* Rule 3  — No REST/HTTP in the execution path (WebSocket/FIX only).
* Rule 4  — No supervised direction/regime label as a training target.
* Rule 5  — CAUSALITY (Kairos): the reasoning-facing bridge (microstructure
            tools + analyst) may read perception ONLY through the causal
            accessors ``as_of`` / ``window_before`` / ``aggregate_before``.
            Touching ``.latest`` / ``._percepts`` / ``._ts`` there would let a
            dated query reach a future percept — the exact look-ahead hole
            Kairos exists to close. Flagged as a violation.

Usage:  python scripts/soul_check.py [--root DIR] [--quiet]
Exit:   0 = clean, 1 = violation(s) found.
"""
from __future__ import annotations

import io
import re
import sys
import token
import tokenize
from pathlib import Path

# --- Rule 2: banned technical-analysis vocabulary ---------------------------
BANNED_TOKENS = {
    "rsi", "macd", "bollinger", "stochastic", "ichimoku", "adx", "obv",
    "mfi", "cci", "vwap", "twap", "ema", "sma", "wma", "dema", "tema",
    "supertrend", "sar", "aroon", "atr", "roc", "ppo", "trix", "kdj",
}
BANNED_PHRASES = [
    r"moving\s*average", r"exponential\s*moving", r"relative\s*strength",
    r"bollinger\s*band", r"parabolic\s*sar", r"price\s*indicator",
]

SCAN_DIRS = ("src", "tests", "scripts")
PY_SUFFIX = {".py"}
CPP_SUFFIX = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx"}
SKIP_PARTS = {".venv", ".git", "build", "__pycache__", "node_modules", "artifacts", "data"}
SELF_NAME = "soul_check.py"

_SUBTOK = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_WORD = re.compile(r"\b(" + "|".join(sorted(BANNED_TOKENS, key=len, reverse=True)) + r")\b", re.I)
_PHRASE = re.compile("|".join(BANNED_PHRASES), re.I)

# Rule 5: forbidden non-causal accessors inside the reasoning-facing bridge.
_NONCAUSAL = re.compile(r"\.(latest\b|_percepts\b|_ts\b)")
_BRIDGE_CAUSAL_FILES = ("bridge/microstructure_tools.py", "bridge/microstructure_analyst.py")


class Violation:
    __slots__ = ("rule", "path", "line", "msg")

    def __init__(self, rule, path, line, msg):
        self.rule, self.path, self.line, self.msg = rule, path, line, msg

    def __str__(self):
        return f"  [Rule {self.rule}] {self.path}:{self.line}  {self.msg}"


def _norm(rel: str) -> str:
    return "/" + rel.replace("\\", "/")


def is_reasoning(relp: str) -> bool:
    """System-2 subtree — exempt from the engine soul (different philosophy)."""
    return "/reasoning/" in relp or "/reasoning_cli/" in relp or "/tests/reasoning/" in relp


def is_engine(relp: str) -> bool:
    """System-1 engine + causal bridge — subject to Rules 1/2/4."""
    if is_reasoning(relp):
        return False
    return (
        "/perception/" in relp or "/bridge/" in relp
        or relp.startswith("/src/cpp") or relp.startswith("/src/bindings")
        or "/tests/perception/" in relp or "/tests/bridge/" in relp or "/tests/loop/" in relp
    )


def _subtokens(identifier: str) -> set[str]:
    return {m.group(0).lower() for m in _SUBTOK.finditer(identifier)}


def scan_python(path: Path, rel: str) -> list[Violation]:
    out: list[Violation] = []
    src = path.read_text(encoding="utf-8", errors="replace")
    relp = _norm(rel)
    engine = is_engine(relp)
    in_exec_path = engine and ("/execution/" in relp or "/ingest/" in relp or "execution_link" in relp)
    in_model_path = engine and any(p in relp for p in ("/models/", "/strategy/"))

    # Rule 2 (engine only) — banned indicator vocabulary in identifiers/strings.
    if engine:
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        except tokenize.TokenError:
            toks = []
        for tok in toks:
            if tok.type == token.NAME:
                hit = _subtokens(tok.string) & BANNED_TOKENS
                if hit:
                    out.append(Violation(2, rel, tok.start[0],
                                         f"identifier '{tok.string}' contains banned indicator {sorted(hit)}"))
            elif tok.type in (token.STRING, tokenize.COMMENT, getattr(token, "FSTRING_MIDDLE", -1)):
                low = tok.string.lower()
                m = _WORD.search(low) or _PHRASE.search(low)
                if m:
                    out.append(Violation(2, rel, tok.start[0],
                                         f"text mentions banned indicator '{m.group(0)}'"))

    # Rule 3 — no REST import in the execution path.
    if in_exec_path:
        for i, ln in enumerate(src.splitlines(), 1):
            if re.search(r"\b(import\s+requests|from\s+requests|import\s+http\.client|import\s+urllib)\b", ln):
                out.append(Violation(3, rel, i,
                                     "REST/HTTP client imported in execution path (use WebSocket/FIX)"))

    # Rule 4 — no supervised direction/regime label in model or strategy code.
    if in_model_path and "regime" in src.lower():
        for i, ln in enumerate(src.splitlines(), 1):
            low = ln.lower()
            if "regime" in low and re.search(r"cross[\s_]?entropy|labels?\s*=|y_true|softmax_cross|classification", low):
                out.append(Violation(4, rel, i,
                                     "ground-truth 'regime' used as a supervised target (label-free only)"))

    # Rule 5 (Kairos causality) — the reasoning-facing bridge must read the bus
    # only through the causal accessors.
    if any(relp.endswith(f) for f in _BRIDGE_CAUSAL_FILES):
        for i, ln in enumerate(src.splitlines(), 1):
            code = ln.split("#", 1)[0]
            m = _NONCAUSAL.search(code)
            if m:
                out.append(Violation(5, rel, i,
                                     f"non-causal bus access '{m.group(0)}' in a reasoning-facing tool "
                                     "(use as_of/window_before/aggregate_before only)"))
    return out


def scan_cpp(path: Path, rel: str) -> list[Violation]:
    out: list[Violation] = []
    relp = _norm(rel)
    if not is_engine(relp):
        return out
    for i, ln in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        code = ln.split("//", 1)[0]
        m = _WORD.search(code) or _PHRASE.search(code)
        if m:
            out.append(Violation(2, rel, i, f"banned indicator '{m.group(0)}'"))
        if re.search(r"\b(memcpy|memmove)\b", code) and "soul:allow-memcpy" not in ln:
            out.append(Violation(1, rel, i,
                                 "memcpy/memmove on LOB path (zero-copy only; waive with 'soul:allow-memcpy')"))
    return out


def iter_sources(root: Path):
    for d in SCAN_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or any(part in SKIP_PARTS for part in p.parts):
                continue
            if p.name == SELF_NAME:
                continue
            if p.suffix in PY_SUFFIX or p.suffix in CPP_SUFFIX:
                yield p


def main(argv: list[str]) -> int:
    root = Path.cwd()
    quiet = "--quiet" in argv
    if "--root" in argv:
        root = Path(argv[argv.index("--root") + 1]).resolve()

    violations: list[Violation] = []
    n_files = n_engine = 0
    for p in iter_sources(root):
        n_files += 1
        rel = str(p.relative_to(root))
        if is_engine(_norm(rel)):
            n_engine += 1
        violations += (scan_python if p.suffix in PY_SUFFIX else scan_cpp)(p, rel)

    if not quiet:
        print(f"soul_check: scanned {n_files} source file(s) ({n_engine} engine/bridge) under {root}")
    if violations:
        print(f"\n\033[31m✗ CONSTITUTION VIOLATED — {len(violations)} issue(s):\033[0m")
        for v in violations:
            print(str(v))
        print("\nThe soul of Kairos is non-negotiable. Fix or rollback (git reset --hard).")
        return 1
    if not quiet:
        print("\033[32m✓ soul intact — no constitutional violations.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
