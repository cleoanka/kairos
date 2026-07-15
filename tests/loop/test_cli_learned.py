"""`kairos loop --learned` fails cleanly when the encoder artifacts are absent.

The trained System-1 encoder (``artifacts/lob_encoder.safetensors`` + friends)
is gitignored and only producible on Apple Silicon + MLX, so a fresh clone has
no artifacts. The flag must then print an actionable message and exit non-zero —
never a bare FileNotFoundError traceback.
"""
from __future__ import annotations

from unittest import mock

import pytest

from kairos import cli


def test_learned_missing_artifact_is_friendly(capsys):
    exc = FileNotFoundError(2, "No such file or directory", "artifacts/latents.npz")
    with mock.patch("kairos.perception.regime.predict.RegimePredictor.load", side_effect=exc):
        rc = cli.main(["loop", "--learned", "--steps", "50"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "kairos perceive --mode synthetic" in err
    assert "MLX" in err and "Apple Silicon" in err


def test_loop_without_learned_never_loads_backend():
    """The friendly guard must not fire on the default (deterministic) path."""
    with mock.patch("kairos.cli._load_learned_backend") as load, \
            mock.patch("kairos.loop.run_cognitive_loop") as run:
        run.return_value.reflection = "ok"
        cli.main(["loop", "--steps", "50"])
    load.assert_not_called()


def test_learned_propagates_other_errors():
    """Only the missing-artifact case is caught; real bugs still surface."""
    with mock.patch("kairos.perception.regime.predict.RegimePredictor.load",
                    side_effect=ValueError("corrupt weights")):
        with pytest.raises(ValueError):
            cli.main(["loop", "--learned", "--steps", "50"])
