import unittest

import pytest

from kairos.reasoning.agents.utils.agent_utils import build_instrument_context
from kairos.reasoning_cli.utils import normalize_ticker_symbol


@pytest.mark.unit
class TickerSymbolHandlingTests(unittest.TestCase):
    def test_normalize_ticker_symbol_preserves_exchange_suffix(self):
        self.assertEqual(normalize_ticker_symbol(" cnc.to "), "CNC.TO")

    def test_build_instrument_context_mentions_exact_symbol(self):
        context = build_instrument_context("7203.T")
        self.assertIn("7203.T", context)
        self.assertIn("exchange suffix", context)

    def test_single_get_ticker_no_shadow(self):
        # Regression: cli/main.py had a duplicate get_ticker with an empty
        # questionary prompt (rendered as a bare "?") that shadowed the
        # descriptive one in cli/utils. Keep a single canonical definition.
        import kairos.reasoning_cli.main
        import kairos.reasoning_cli.utils
        self.assertIs(kairos.reasoning_cli.main.get_ticker, kairos.reasoning_cli.utils.get_ticker)


if __name__ == "__main__":
    unittest.main()
