"""StockTwits fetch degrades (never raises) on transport errors, including the
http.client chunked-transfer exceptions that are not OSErrors (#1024)."""

from __future__ import annotations

import http.client
import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from kairos.reasoning.dataflows import stocktwits


def _raise(exc):
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner):
            raise exc
    return _Resp()


def _payload(data):
    body = json.dumps(data).encode()

    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner):
            return body
    return _Resp()


@pytest.mark.unit
class StockTwitsResilienceTests:
    @pytest.mark.parametrize(
        "exc",
        [
            http.client.IncompleteRead(b""),
            HTTPError("url", 503, "down", {}, None),
            TimeoutError("slow"),
        ],
    )
    def test_transport_errors_return_placeholder(self, exc):
        with patch.object(stocktwits, "urlopen", return_value=_raise(exc)):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert "unavailable" in out.lower()
        assert out.startswith("<stocktwits unavailable")


@pytest.mark.unit
class TestStockTwitsSymbolValidation:
    """A malformed/injection ticker must be rejected before it reaches the URL
    path, degrading to the placeholder rather than issuing the request."""

    @pytest.mark.parametrize(
        "ticker",
        ["NVDA/../admin", "AAPL foo", "AAPL?x=1", "AA&PL", ""],
    )
    def test_invalid_ticker_returns_placeholder_without_request(self, ticker):
        with patch.object(
            stocktwits, "urlopen",
            side_effect=AssertionError("network must not be reached"),
        ):
            out = stocktwits.fetch_stocktwits_messages(ticker)
        assert out.startswith("<stocktwits unavailable: invalid ticker")


@pytest.mark.unit
class TestStockTwitsMalformedMessage:
    """A non-dict element in the ``messages`` array (null/list/str) must be
    skipped rather than raising an AttributeError, so the well-formed siblings
    still render and the documented contract holds."""

    @pytest.mark.parametrize("bad", [None, ["not", "a", "dict"], "raw string", 42])
    def test_non_dict_message_element_is_skipped(self, bad):
        payload = {
            "messages": [
                bad,
                {
                    "created_at": "2026-07-15T00:00:00Z",
                    "user": {"username": "trader"},
                    "entities": {"sentiment": {"basic": "Bullish"}},
                    "body": "clean message",
                },
            ]
        }
        with patch.object(stocktwits, "urlopen", return_value=_payload(payload)):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        # The good message survives; the malformed one is dropped, not counted.
        assert "@trader" in out
        assert "clean message" in out
        assert "Total: 1 most-recent messages" in out
