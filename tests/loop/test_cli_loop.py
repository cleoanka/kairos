"""`kairos loop` fails cleanly on out-of-contract flags — never a raw traceback.

Two robustness contracts: (a) a --steps/--decision-fraction combination that
leaves no forward window past ``perception_window`` must print an actionable
message and exit non-zero rather than surfacing the bare ``ValueError`` from
``run_cognitive_loop``; (b) --decision-fraction must be a genuine fraction in
``(0, 1)`` — a value ``<= 0`` or ``>= 1`` silently runs with wrong semantics
(the decision point clamps to ``perception_window``), so it is rejected up front.
"""
from __future__ import annotations

import pytest

from kairos import cli


def test_no_forward_window_is_friendly(capsys):
    """--steps below perception_window leaves no forward window: clean exit, no traceback."""
    rc = cli.main(["loop", "--steps", "10"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no forward window" in err
    assert "--steps" in err and "perception_window" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("frac", ["0", "-0.5", "1", "1.5"])
def test_decision_fraction_out_of_bounds_rejected(capsys, frac):
    """A --decision-fraction outside (0, 1) is rejected before the loop runs."""
    rc = cli.main(["loop", "--scenario", "calm", "--steps", "200", "--decision-fraction", frac])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--decision-fraction must be in (0, 1)" in err
    assert frac in err


def test_decision_fraction_in_bounds_runs(capsys):
    """A valid fraction still runs the loop and prints the reflection (no regression)."""
    rc = cli.main(["loop", "--scenario", "calm", "--steps", "200", "--decision-fraction", "0.5"])
    assert rc == 0
    assert "Edge vs stand-aside" in capsys.readouterr().out
