"""API keys must never survive into an error string.

Alpha Vantage and FRED pass the key as a URL query param, so a raw ``requests``
exception echoes it in its message — which then flows into logs *and* the
agent-facing NO_DATA/DATA_UNAVAILABLE text. These tests pin the source-level
redaction that closes that leak.
"""
import pytest

import kairos.reasoning.dataflows.fred as fred
from kairos.reasoning.dataflows._redact import redact_secrets


@pytest.mark.unit
def test_redact_secrets_masks_known_value_and_query_params():
    key = "SUPERSECRET_KEY_123"
    text = (
        "ConnectionError: url: /query?function=TIME_SERIES&apikey=SUPERSECRET_KEY_123"
        "&api_key=abc123&token=zzz&other=keepme"
    )
    out = redact_secrets(text, key)
    assert key not in out
    assert "abc123" not in out       # generic api_key= param masked even without the literal
    assert "zzz" not in out          # token= masked
    assert "keepme" in out           # non-secret params untouched
    assert "apikey=***" in out


@pytest.mark.unit
def test_redact_secrets_handles_none_and_non_str():
    assert redact_secrets(12345) == "12345"
    assert redact_secrets("x", None, "") == "x"   # empty/None extra values are skipped


@pytest.mark.unit
def test_fred_request_redacts_key_on_network_error(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "FRED_SECRET_XYZ")

    def boom(url, params=None, **kwargs):
        raise fred.requests.exceptions.ConnectionError(
            "HTTPSConnectionPool(host='api.stlouisfed.org'): Max retries exceeded "
            "with url: /fred/series?series_id=CPIAUCSL&api_key=FRED_SECRET_XYZ&file_type=json"
        )

    monkeypatch.setattr(fred.requests, "get", boom)
    with pytest.raises(fred.FredRequestError) as ei:
        fred._request("series", {"series_id": "CPIAUCSL"})
    msg = str(ei.value)
    assert "FRED_SECRET_XYZ" not in msg
    assert "api_key=***" in msg
