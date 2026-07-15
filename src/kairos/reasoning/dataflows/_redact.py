"""Secret redaction for vendor error text.

Vendors like Alpha Vantage and FRED carry the API key as a URL query parameter
(``?...&apikey=SECRET``). When ``requests`` raises a ``ConnectionError`` /
``Timeout`` / ``HTTPError``, its message echoes the **full URL** — key included.
That string then flows into ``logger.warning(..., e)`` *and*, worse, into the
agent-facing ``NO_DATA`` / ``DATA_UNAVAILABLE`` text, exposing the secret in
logs and in the LLM context. :func:`redact_secrets` scrubs it at the source so
no downstream sink ever sees the key.
"""

from __future__ import annotations

import re

# Query-string secrets: apikey=..., api_key=..., token=..., access_token=...,
# secret=..., password=... — value runs until the next '&', whitespace or quote.
_SECRET_QS = re.compile(
    r"(?i)\b(api[_-]?key|apikey|token|access[_-]?token|secret|password)=[^&\s'\"]+"
)


def redact_secrets(text: object, *values: str | None) -> str:
    """Stringify ``text`` and mask any secrets in it.

    Every non-empty literal in ``values`` (e.g. the actual API key) is replaced
    first — the most reliable scrub, since we hold the exact value — then a
    generic query-string pattern catches key-bearing params whose value we do
    not have on hand.
    """
    s = str(text)
    for v in values:
        if v:
            s = s.replace(v, "***")
    return _SECRET_QS.sub(r"\1=***", s)
