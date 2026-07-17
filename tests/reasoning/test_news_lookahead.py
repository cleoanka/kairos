"""yfinance news must not leak future-dated (or undated, in a backtest) articles
into a historical window.

Regressions for #992 (flat articles bypassed the date filter), #1007 (global
news injected future articles), #993 (empty-after-filter returned a blank body).
"""
import calendar
import time
from datetime import datetime

import pytest

import kairos.reasoning.dataflows.yfinance_news as ynews


def _epoch(date_str, hour=0):
    # providerPublishTime is a UTC UNIX epoch — pin the frame with calendar.timegm
    # (UTC), not time.mktime (LOCAL), so the flat/nested TZ divergence is exercised
    # (a LOCAL epoch would cancel against a LOCAL flat parse and hide the leak).
    return calendar.timegm(datetime(*map(int, date_str.split("-")), hour).timetuple())


@pytest.mark.unit
def test_flat_article_publish_time_is_parsed():
    # #992: flat articles now carry a pub_date (was always None -> unfilterable).
    data = ynews._extract_article_data(
        {"title": "X", "publisher": "P", "link": "l", "providerPublishTime": _epoch("2025-05-09")}
    )
    assert data["pub_date"] is not None
    assert data["pub_date"].strftime("%Y-%m-%d") == "2025-05-09"


@pytest.mark.unit
def test_flat_publish_time_is_utc_not_host_local(monkeypatch):
    # Flat and nested paths must land on the SAME (UTC) frame. An article
    # published 2024-01-11 02:00 UTC sits *past* an end_date=2024-01-10 window
    # (upper bound 2024-01-11 00:00 UTC), so it must be excluded on every host.
    # Under the old LOCAL parse, a host behind UTC (e.g. America/Los_Angeles)
    # would read it as 2024-01-10 18:00 and leak it — the #992/#1007 look-ahead.
    if not hasattr(time, "tzset"):  # pragma: no cover - Windows has no tzset
        pytest.skip("tzset unavailable on this platform")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        data = ynews._extract_article_data(
            {"title": "POST", "publisher": "P", "link": "l",
             "providerPublishTime": _epoch("2024-01-11", hour=2)}
        )
        # UTC parse preserves the true 02:00 UTC wall-clock (not shifted to local).
        assert data["pub_date"].strftime("%Y-%m-%d %H") == "2024-01-11 02"
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)
        assert ynews._in_news_window(data["pub_date"], start, end) is False  # leak blocked
    finally:
        monkeypatch.delenv("TZ", raising=False)
        time.tzset()


@pytest.mark.unit
def test_window_excludes_future_and_undated_in_backtest():
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)  # historical window (well in the past)
    inside = datetime(2025, 5, 5)
    future = datetime(2025, 6, 1)
    assert ynews._in_news_window(inside, start, end) is True
    assert ynews._in_news_window(future, start, end) is False     # look-ahead blocked
    assert ynews._in_news_window(None, start, end) is False        # undated -> excluded in backtest


@pytest.mark.unit
def test_window_keeps_undated_in_live_window():
    # Live window (reaches today): undated articles can't be "future", so keep them.
    start = datetime.now()
    end = datetime.now()
    assert ynews._in_news_window(None, start, end) is True


@pytest.mark.unit
def test_global_news_future_flat_article_excluded(monkeypatch):
    # #1007: a flat, future-dated global article must not appear in a historical run.
    future_article = {"title": "FUTURE EVENT", "publisher": "P", "link": "l",
                      "providerPublishTime": _epoch("2025-06-01")}
    past_article = {"title": "PAST EVENT", "publisher": "P", "link": "l",
                    "providerPublishTime": _epoch("2025-05-05")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [future_article, past_article]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "PAST EVENT" in out
    assert "FUTURE EVENT" not in out  # #1007


@pytest.mark.unit
def test_global_news_empty_after_filter_is_informative(monkeypatch):
    # #993: everything filtered out -> a clear message, not a blank-bodied report.
    only_future = {"title": "FUTURE", "publisher": "P", "link": "l",
                   "providerPublishTime": _epoch("2025-06-01")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [only_future]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "No global news found" in out
    assert "###" not in out  # no empty article body
