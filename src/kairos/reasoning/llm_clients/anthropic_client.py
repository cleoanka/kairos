import re
import warnings
from typing import Any

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens", "temperature",
    "callbacks", "http_client", "http_async_client", "effort",
)

# Anthropic's extended-thinking ``effort`` parameter is accepted by Opus 4.5+
# and Sonnet 4.6+ only. Sonnet 4.5 and any Haiku version 400 with
# ``"This model does not support the effort parameter"`` (#831). The per-family
# minimum version below is forward-compatible: future ``claude-{opus,sonnet}-X-Y``
# releases inherit support automatically, while Sonnet 4.5 and Haiku stay excluded.
_EFFORT_EXACT = {
    "claude-mythos-preview",  # non-standard preview name; effort-capable
}
_EFFORT_MODEL = re.compile(r"^claude-(opus|sonnet)-(\d+)-(\d+)$")
_EFFORT_MIN_VERSION = {"opus": (4, 5), "sonnet": (4, 6)}


def _supports_effort(model: str) -> bool:
    """Whether Anthropic accepts the ``effort`` parameter for this model."""
    model_lc = model.lower()
    if model_lc in _EFFORT_EXACT:
        return True
    match = _EFFORT_MODEL.match(model_lc)
    if not match:
        return False
    family, major, minor = match.group(1), int(match.group(2)), int(match.group(3))
    return (major, minor) >= _EFFORT_MIN_VERSION[family]


# Opus 4.7 *removed* the sampling parameters: temperature / top_p / top_k are
# rejected with a 400 on Opus 4.7+ **unconditionally** — whether or not
# ``effort`` is sent. This is a MODEL-based rule, not an effort-based one: Opus
# 4.5/4.6 and every Sonnet still accept temperature (even alongside effort). The
# Fable-5 family (and its ``claude-mythos-preview`` predecessor) shares that API
# surface and also removes sampling params, so it is special-cased like the
# effort gate's ``_EFFORT_EXACT``.
_SAMPLING_REMOVED = re.compile(r"^claude-opus-(\d+)-(\d+)$")
_SAMPLING_REMOVED_MIN = (4, 7)
_SAMPLING_REMOVED_EXACT = {"claude-mythos-preview"}


def _rejects_sampling_params(model: str) -> bool:
    """Whether Anthropic 400s on temperature/top_p/top_k for this model."""
    model_lc = model.lower()
    if model_lc in _SAMPLING_REMOVED_EXACT:
        return True
    match = _SAMPLING_REMOVED.match(model_lc)
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= _SAMPLING_REMOVED_MIN


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        rejects_sampling = _rejects_sampling_params(self.model)

        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "effort" and not _supports_effort(self.model):
                continue
            # Opus 4.7+ REMOVED the sampling params: ``temperature`` 400s on those
            # models regardless of whether ``effort`` is set. Drop it for exactly
            # those models (Opus 4.5/4.6 and Sonnet still accept temperature). The
            # gate is model-based, not effort-based (kai6-00 fixed the round-5
            # effort-based version, which both missed Opus 4.7+temperature-no-effort
            # and wrongly dropped temperature on Opus 4.5/4.6).
            if key == "temperature" and rejects_sampling:
                warnings.warn(
                    f"Dropping 'temperature' for model '{self.model}': Opus 4.7+ "
                    f"rejects sampling parameters (temperature/top_p/top_k).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
