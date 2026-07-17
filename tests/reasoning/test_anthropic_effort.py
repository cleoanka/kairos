"""Tests for Anthropic effort-parameter gating (#831).

Haiku (any version) and Sonnet 4.5 reject the ``effort`` parameter with a
400. Only Opus 4.5+ and Sonnet 4.6+ accept it. The gate uses a per-family
minimum version so future ``claude-{opus,sonnet}-X-Y`` releases inherit
support automatically.
"""

import pytest

from kairos.reasoning.llm_clients import anthropic_client as mod


def _capture_kwargs(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        mod, "NormalizedChatAnthropic",
        lambda **kwargs: captured.setdefault("kwargs", kwargs),
    )
    return captured


@pytest.mark.unit
class TestEffortGate:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-haiku-4-5", "claude-haiku-5-0", "claude-haiku-4-7-preview",
            # Sonnet 4.5 (and earlier) 400 on effort — only Sonnet 4.6+ supports it.
            "claude-sonnet-4-5", "claude-sonnet-4-0",
        ],
    )
    def test_unsupported_models_do_not_receive_effort(self, monkeypatch, model):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(model=model, effort="medium", api_key="x").get_llm()
        assert "effort" not in captured["kwargs"]

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7",
            "claude-sonnet-4-6",
        ],
    )
    def test_current_opus_and_sonnet_receive_effort(self, monkeypatch, model):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(model=model, effort="high", api_key="x").get_llm()
        assert captured["kwargs"]["effort"] == "high"

    @pytest.mark.parametrize(
        "model",
        ["claude-opus-5-0", "claude-opus-4-8", "claude-sonnet-5-0"],
    )
    def test_future_opus_sonnet_inherit_effort_via_pattern(self, monkeypatch, model):
        """Forward-compat: new Opus/Sonnet versions don't need a code change."""
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(model=model, effort="low", api_key="x").get_llm()
        assert captured["kwargs"]["effort"] == "low"

    def test_mythos_preview_receives_effort(self, monkeypatch):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(
            model="claude-mythos-preview", effort="medium", api_key="x"
        ).get_llm()
        assert captured["kwargs"]["effort"] == "medium"

    def test_unknown_anthropic_model_does_not_receive_effort(self, monkeypatch):
        """Default is conservative — unknown models don't get effort to avoid 400s."""
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(
            model="claude-experimental-x", effort="medium", api_key="x"
        ).get_llm()
        assert "effort" not in captured["kwargs"]

    def test_other_kwargs_still_forwarded_when_effort_skipped(self, monkeypatch):
        """Skipping effort must not break other passthrough kwargs."""
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(
            model="claude-haiku-4-5",
            effort="medium",
            api_key="placeholder",
            max_tokens=1024,
            timeout=30,
        ).get_llm()
        assert captured["kwargs"]["api_key"] == "placeholder"
        assert captured["kwargs"]["max_tokens"] == 1024
        assert captured["kwargs"]["timeout"] == 30
        assert "effort" not in captured["kwargs"]


@pytest.mark.unit
class TestTemperatureEffortMutualExclusion:
    """``temperature`` and ``effort`` are mutually exclusive per the Anthropic API.

    Effort-capable models (Opus 4.7+ / Fable 5) REMOVE ``temperature`` and 400
    when both are sent. ``get_llm`` must drop temperature when effort is active,
    mirroring the effort gate (kai5-02).
    """

    def test_effort_capable_model_does_not_receive_temperature(self, monkeypatch):
        captured = _capture_kwargs(monkeypatch)
        with pytest.warns(RuntimeWarning, match="temperature"):
            mod.AnthropicClient(
                model="claude-opus-4-8", effort="high", temperature=0.2, api_key="x"
            ).get_llm()
        assert "temperature" not in captured["kwargs"]
        assert captured["kwargs"]["effort"] == "high"

    def test_temperature_kept_when_no_effort(self, monkeypatch):
        """Without effort, temperature passes through unchanged."""
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(
            model="claude-opus-4-8", temperature=0.2, api_key="x"
        ).get_llm()
        assert captured["kwargs"]["temperature"] == 0.2

    def test_temperature_kept_when_effort_unsupported(self, monkeypatch):
        """A model that ignores effort still accepts temperature — keep it."""
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(
            model="claude-haiku-4-5", effort="high", temperature=0.2, api_key="x"
        ).get_llm()
        assert captured["kwargs"]["temperature"] == 0.2
        assert "effort" not in captured["kwargs"]
