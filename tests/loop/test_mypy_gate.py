"""The scoped mypy type gate stays green over the Kairos-original core.

`pyproject.toml` gates mypy on exactly the causal bridge, the cognitive loop,
and the CLI — the neuro-symbolic core that must be type-sound — while leaving
the vendored TradingAgents tree (`kairos.reasoning*`, a long tail of upstream
typing debt) out of the gate on purpose. This is the executable half of that
config: (a) the gate is wired to those and only those paths, and (b) it runs
clean today, so any type regression that lands there turns CI red.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

_ROOT = Path(__file__).resolve().parents[2]


def _mypy_config() -> dict:
    cfg = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    return cfg.get("tool", {}).get("mypy", {})


def test_gate_scopes_only_the_kairos_original_core():
    """The gate covers bridge + loop + cli.py and never the vendored reasoning tree."""
    files = _mypy_config().get("files", [])
    assert files == ["src/kairos/bridge", "src/kairos/loop", "src/kairos/cli.py"]
    assert not any("reasoning" in f for f in files)   # vendored long tail stays out


def test_gate_checks_untyped_bodies_with_missing_imports_ignored():
    """The gate has teeth (untyped bodies checked) yet tolerates un-stubbed deps."""
    cfg = _mypy_config()
    assert cfg.get("ignore_missing_imports") is True
    assert cfg.get("check_untyped_defs") is True
    assert cfg.get("mypy_path") == "src"


def test_mypy_passes_on_the_gated_paths():
    """`mypy` (reading pyproject) reports zero errors over the gated subtree."""
    pytest.importorskip("mypy")
    proc = subprocess.run(
        [sys.executable, "-m", "mypy"],
        cwd=_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"mypy gate is red:\n{proc.stdout}{proc.stderr}"
    assert "Success" in proc.stdout
