"""Alpha Vantage request hardening.

Regressions for #990 (no request timeout -> can hang), #991 (invalid-key
responses mislabeled as rate limits and silently treated as transient), the
date-filter fail-open look-ahead leak, and the API-key-in-error-log leak.
"""
import pytest

import kairos.reasoning.dataflows.alpha_vantage_common as av
from kairos.reasoning.dataflows.errors import NoMarketDataError


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def _patched_get(body, capture=None):
    def fake_get(url, params=None, **kwargs):
        if capture is not None:
            capture.update(kwargs)
        return _FakeResponse(body)
    return fake_get


@pytest.mark.unit
def test_request_passes_timeout(monkeypatch):
    captured = {}
    monkeypatch.setattr(av.requests, "get", _patched_get("Date,Close\n2025-01-02,1.0", captured))
    av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    assert captured.get("timeout") == av.REQUEST_TIMEOUT  # #990


@pytest.mark.unit
def test_rate_limit_detected(monkeypatch):
    body = '{"Information": "Our standard API rate limit is 25 requests per day. ... your API key ..."}'
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageRateLimitError):
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})


@pytest.mark.unit
def test_invalid_key_not_mislabeled_as_rate_limit(monkeypatch):
    # AV's invalid-key notice mentions "API key"; it must NOT be treated as a
    # (transient) rate limit, but surface as a real configuration error (#991).
    body = ('{"Information": "the parameter apikey is invalid or missing. '
            'Please claim your free API key on (https://www.alphavantage.co/support/#api-key)."}')
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageNotConfiguredError):
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    with pytest.raises(av.AlphaVantageRateLimitError):  # sanity: rate-limit path still distinct
        monkeypatch.setattr(av.requests, "get", _patched_get('{"Note": "API call frequency is 5 calls per minute."}'))
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})


# --- Date-filter must FAIL CLOSED, never serve unfiltered (future) data ------

@pytest.mark.unit
def test_filter_trims_future_and_out_of_range_rows():
    csv = "date,close\n2024-04-01,50\n2024-05-10,100\n2024-06-15,200\n"
    out = av._filter_csv_by_date_range(csv, "2024-05-01", "2024-05-31", symbol="X")
    assert "2024-05-10" in out
    assert "2024-06-15" not in out   # future row (after end_date) trimmed — the whole point
    assert "2024-04-01" not in out   # before start_date trimmed


@pytest.mark.unit
def test_filter_all_unparseable_dates_fails_closed():
    # First column isn't a date at all -> ordering can't be trusted -> returning
    # it unfiltered would leak future rows. Must raise NoMarketDataError.
    csv = "sym,close\nAAA,1\nBBB,2\n"
    with pytest.raises(NoMarketDataError):
        av._filter_csv_by_date_range(csv, "2024-01-01", "2024-06-01", symbol="X")


@pytest.mark.unit
def test_filter_structural_failure_fails_closed(monkeypatch):
    # A raw parse blow-up must NOT fall back to the unfiltered response.
    def boom(*a, **k):
        raise ValueError("corrupt frame")
    monkeypatch.setattr(av.pd, "read_csv", boom)
    with pytest.raises(NoMarketDataError):
        av._filter_csv_by_date_range(
            "date,close\n2099-01-01,999\n", "2024-01-01", "2024-06-01", symbol="X"
        )


@pytest.mark.unit
def test_filter_drops_only_the_unparseable_rows():
    # A few bad rows are dropped (a NaT date can't be proven in-range) while the
    # good, in-range rows survive — partial corruption doesn't sink the frame.
    csv = "date,close\n2024-05-10,100\nGARBAGE,200\n"
    out = av._filter_csv_by_date_range(csv, "2024-05-01", "2024-05-31", symbol="X")
    assert "2024-05-10" in out
    assert "200" not in out


# --- API key must never leak into an error string (#log/#agent-channel) ------

@pytest.mark.unit
def test_api_key_redacted_on_network_error(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "SUPERSECRET_KEY_123")

    def boom(url, params=None, **kwargs):
        # requests echoes the full request URL — key included — in its message.
        raise av.requests.exceptions.ConnectionError(
            "HTTPSConnectionPool(host='www.alphavantage.co', port=443): Max retries "
            "exceeded with url: /query?function=TIME_SERIES_DAILY_ADJUSTED&"
            "apikey=SUPERSECRET_KEY_123 (Caused by NewConnectionError(...))"
        )

    monkeypatch.setattr(av.requests, "get", boom)
    with pytest.raises(av.AlphaVantageRequestError) as ei:
        av._make_api_request("TIME_SERIES_DAILY_ADJUSTED", {"symbol": "AAPL"})
    msg = str(ei.value)
    assert "SUPERSECRET_KEY_123" not in msg    # the raw key is gone
    assert "apikey=***" in msg                 # redacted, but the shape is still legible
