import json
import os
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from ._redact import redact_secrets
from .errors import NoMarketDataError, VendorNotConfiguredError, VendorRateLimitError

API_BASE_URL = "https://www.alphavantage.co/query"

# Network timeout (seconds) so a stalled Alpha Vantage request can't hang the
# CLI/agents indefinitely (#990).
REQUEST_TIMEOUT = 30


class AlphaVantageNotConfiguredError(VendorNotConfiguredError):
    """Raised when Alpha Vantage is selected but no API key is configured.

    A VendorNotConfiguredError (and thus still a ValueError), so the routing
    layer's "vendor unavailable" handling and existing ValueError callers both
    keep working.
    """
    pass


def get_api_key() -> str:
    """Retrieve the API key for Alpha Vantage from environment variables."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise AlphaVantageNotConfiguredError(
            "ALPHA_VANTAGE_API_KEY environment variable is not set."
        )
    return api_key

def format_datetime_for_api(date_input) -> str:
    """Convert various date formats to YYYYMMDDTHHMM format required by Alpha Vantage API."""
    if isinstance(date_input, str):
        # If already in correct format, return as-is
        if len(date_input) == 13 and 'T' in date_input:
            return date_input
        # Try to parse common date formats
        try:
            dt = datetime.strptime(date_input, "%Y-%m-%d")
            return dt.strftime("%Y%m%dT0000")
        except ValueError:
            try:
                dt = datetime.strptime(date_input, "%Y-%m-%d %H:%M")
                return dt.strftime("%Y%m%dT%H%M")
            except ValueError:
                raise ValueError(f"Unsupported date format: {date_input}") from None
    elif isinstance(date_input, datetime):
        return date_input.strftime("%Y%m%dT%H%M")
    else:
        raise ValueError(f"Date must be string or datetime object, got {type(date_input)}")

class AlphaVantageRateLimitError(VendorRateLimitError):
    """Raised when the Alpha Vantage API rate limit is exceeded."""
    pass

class AlphaVantageRequestError(Exception):
    """A network/HTTP failure talking to Alpha Vantage, with the API key redacted.

    A raw ``requests`` exception echoes the full request URL — including
    ``apikey=...`` — in its message. We re-raise this key-free type so the router
    can still surface a broken primary vendor (#989) without leaking the secret
    into logs or the agent-facing data channel.
    """
    pass

def _make_api_request(function_name: str, params: dict) -> dict | str:
    """Helper function to make API requests and handle responses.

    Raises:
        AlphaVantageRateLimitError: When API rate limit is exceeded
    """
    # Create a copy of params to avoid modifying the original
    api_params = params.copy()
    api_key = get_api_key()
    api_params.update({
        "function": function_name,
        "apikey": api_key,
        "source": "trading_agents",
    })

    # Handle entitlement parameter if present in params or global variable
    current_entitlement = globals().get('_current_entitlement')
    entitlement = api_params.get("entitlement") or current_entitlement

    if entitlement:
        api_params["entitlement"] = entitlement
    elif "entitlement" in api_params:
        # Remove entitlement if it's None or empty
        api_params.pop("entitlement", None)

    try:
        response = requests.get(API_BASE_URL, params=api_params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # The request URL carries apikey=...; a raw requests exception echoes the
        # full URL (key included) in its message and in e.request.url. Redact and
        # re-raise a key-free error before this can reach any log or the agent.
        raise AlphaVantageRequestError(
            redact_secrets(f"{type(e).__name__}: {e}", api_key)
        ) from None

    response_text = response.text

    # Error responses are JSON; data responses are usually CSV (or data-keyed
    # JSON). A non-JSON body is normal data.
    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text

    # Alpha Vantage reports problems via "Information" / "Note". Classify so a
    # genuine rate limit and an invalid/missing key aren't conflated (#991):
    # rate-limit phrasing is checked first because those notices also mention
    # "API key" ("your API key ... 25 requests per day").
    notice = response_json.get("Information") or response_json.get("Note")
    if notice:
        low = notice.lower()
        if any(m in low for m in ("rate limit", "requests per day", "call frequency", "premium")):
            raise AlphaVantageRateLimitError(f"Alpha Vantage rate limit exceeded: {notice}")
        if "api key" in low or "apikey" in low:
            # Reuse the existing "not configured" error so a bad key surfaces as
            # a real, actionable failure rather than a mislabeled rate limit (#991).
            raise AlphaVantageNotConfiguredError(f"Alpha Vantage API key invalid or missing: {notice}")

    return response_text



def _filter_csv_by_date_range(
    csv_data: str, start_date: str, end_date: str, symbol: str = ""
) -> str:
    """Filter CSV rows to ``start_date <= date <= end_date``.

    This is the *only* thing that trims future rows out of the Alpha Vantage
    price series: :func:`get_stock` requests the full series up to *today* with
    no server-side date bound, so a leak here is a leak of look-ahead into a
    causal backtest. The old behaviour returned the **raw, unfiltered** response
    on any parse failure — silently serving every row after ``end_date``. That
    is forbidden: on failure we now **fail closed** (raise
    :class:`NoMarketDataError`) so the router falls back to the next vendor, and
    unparseable individual rows are dropped (a ``NaT`` date can never be proven
    ``<= end_date``) rather than passed through.

    Args:
        csv_data: CSV string from Alpha Vantage.
        start_date / end_date: inclusive bounds, yyyy-mm-dd.
        symbol: for the fail-closed error message.

    Returns:
        The filtered CSV string.

    Raises:
        NoMarketDataError: the frame could not be parsed/filtered safely.
    """
    if not csv_data or csv_data.strip() == "":
        return csv_data

    try:
        df = pd.read_csv(StringIO(csv_data))
        if df.empty or len(df.columns) == 0:
            return csv_data  # header-only / no rows: nothing to leak, nothing to filter

        # Assume the first column is the date column (timestamp). Parse strictly
        # as ISO-8601 (Alpha Vantage's format): deterministic, and it refuses to
        # locale-guess an ambiguous "01/02/2024" the way dateutil would — an
        # unrecognised value becomes NaT and is dropped, never mis-ordered.
        date_col = df.columns[0]
        parsed = pd.to_datetime(df[date_col], errors="coerce", format="ISO8601")
        if int(parsed.isna().sum()) == len(df):
            # EVERY date failed to parse -> the first column is not the date we
            # assumed, so we cannot trust any row's ordering. Returning the frame
            # unfiltered would leak post-end_date rows; fail closed instead.
            raise NoMarketDataError(
                symbol or "?",
                detail=(
                    f"Alpha Vantage response had no parseable dates in its first "
                    f"column ({date_col!r}); refusing to serve it unfiltered."
                ),
            )

        df = df.assign(**{date_col: parsed})
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        # NaT rows compare False against both bounds and are excluded — a row
        # whose date won't parse can never be proven in-range (fail closed).
        mask = (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
        return df[mask].to_csv(index=False)

    except NoMarketDataError:
        raise
    except Exception as e:
        # A structural failure must NEVER fall back to the raw, unfiltered
        # response (every row up to *today* -> look-ahead). Fail closed so the
        # router moves to the next vendor. The pandas exception cannot carry an
        # API key, so echoing its text is safe.
        raise NoMarketDataError(
            symbol or "?",
            detail=(
                f"Alpha Vantage date-range filter failed "
                f"({type(e).__name__}: {e}); refusing to serve unfiltered data."
            ),
        ) from e
