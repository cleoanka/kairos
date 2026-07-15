"""Tests that the yfinance CSV cache is written atomically.

A non-atomic ``to_csv`` straight onto the cache path lets a concurrent reader
observe a half-flushed frame. load_ohlcv must stage the download in a temp file
and os.replace it into place, so a reader interleaved with the write sees either
the previous file or the complete new one — never a truncated CSV.
"""

import os
import unittest
from unittest import mock

import pandas as pd
import pytest

from kairos.reasoning.dataflows import stockstats_utils
from kairos.reasoning.dataflows.config import set_config


def _ohlcv() -> pd.DataFrame:
    dates = pd.bdate_range("2026-04-01", periods=10)
    return pd.DataFrame({
        "Date": dates,
        "Open": [100.0 + i for i in range(10)],
        "High": [101.0 + i for i in range(10)],
        "Low": [99.0 + i for i in range(10)],
        "Close": [100.5 + i for i in range(10)],
        "Volume": [1_000_000 + i for i in range(10)],
    })


@pytest.mark.unit
class TestLoadOhlcvAtomicCache(unittest.TestCase):
    def setUp(self):
        self._tmp = os.path.join(os.path.dirname(__file__), "_tmp_atomic_cache")
        os.makedirs(self._tmp, exist_ok=True)
        set_config({"data_cache_dir": self._tmp})

    def tearDown(self):
        for f in os.listdir(self._tmp):
            os.remove(os.path.join(self._tmp, f))
        os.rmdir(self._tmp)

    def test_cache_written_via_replace_not_direct(self):
        # The final cache path must be produced by os.replace (atomic rename),
        # never by writing DataFrame.to_csv straight onto it — otherwise a
        # concurrent reader can observe a partial file.
        real_replace = os.replace
        replaced = []

        def spy_replace(src, dst):
            replaced.append((src, dst))
            return real_replace(src, dst)

        with mock.patch.object(stockstats_utils.yf, "download", return_value=_ohlcv()), \
                mock.patch.object(stockstats_utils.os, "replace", side_effect=spy_replace):
            stockstats_utils.load_ohlcv("AAPL", "2026-04-10")

        # Exactly one file remains: the cache. No .tmp leftovers.
        files = os.listdir(self._tmp)
        self.assertEqual(len(files), 1)
        self.assertFalse(files[0].endswith(".tmp"))

        # And it landed via os.replace, from a temp file in the same directory.
        self.assertEqual(len(replaced), 1)
        src, dst = replaced[0]
        self.assertEqual(os.path.dirname(src), os.path.dirname(dst))
        self.assertTrue(src.endswith(".tmp"))
        self.assertEqual(dst, os.path.join(self._tmp, files[0]))

    def test_partial_temp_write_never_exposed_at_cache_path(self):
        # Simulate a writer that crashes after staging the temp file but before
        # the rename. The cache path must not exist (nothing partial served);
        # only the abandoned temp file remains for a later run to overwrite.
        boom = RuntimeError("crash mid-write")

        with mock.patch.object(stockstats_utils.yf, "download", return_value=_ohlcv()), \
                mock.patch.object(stockstats_utils.os, "replace", side_effect=boom), \
                self.assertRaises(RuntimeError):
            stockstats_utils.load_ohlcv("AAPL", "2026-04-10")

        cache_file = [f for f in os.listdir(self._tmp) if not f.endswith(".tmp")]
        self.assertEqual(cache_file, [])  # no truncated frame at the cache path


if __name__ == "__main__":
    unittest.main()
