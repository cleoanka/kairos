"""LOB-Core data contract.

This module is the single source of truth for the shape of every Limit Order
Book (LOB) observation that flows through the system: synthetic generator ->
parquet on disk -> MLX feature tensor -> self-supervised model -> regime
clustering. Every other module imports its layout from here so the C++ side and
the Python side agree byte-for-byte.

CONSTITUTION COMPLIANCE
-----------------------
* Rule 2 (no classic analysis): only raw L2 depth, cancel-flow and trade-tick
  fields are defined here. There is intentionally no smoothed, trend-following
  or oscillator-style field anywhere in the contract.
* Rule 4 (label-free learning): a ``regime`` column exists ONLY as
  ground-truth for *evaluation / visualisation*. It is never part of the
  feature matrix and must never enter a training loss. See ``featurize()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

# --- Book geometry -----------------------------------------------------------
N_LEVELS = 10          # depth captured per side (L2 aggregated)
TICK_SCALE = 50.0      # normaliser for price offsets expressed in ticks
SIZE_SCALE = 6.0       # normaliser for log1p(size); log1p(~400) ~= 6
CXL_SCALE = 6.0        # normaliser for log1p(cancel volume)
TRADE_SCALE = 6.0      # normaliser for log1p(|signed trade flow|)


class Regime(IntEnum):
    """Ground-truth market regime — EVALUATION ONLY (never a training target)."""
    RANGE = 0   # balanced two-sided market-maker liquidity
    TREND = 1   # one-sided aggressive retail flow consumes the book
    TOXIC = 2   # spoofer floods/cancels phantom depth (illusory liquidity)


REGIME_NAMES = {r.value: r.name for r in Regime}
REGIME_STYLE = {  # rich-friendly colours for the TUI
    Regime.RANGE: "cyan",
    Regime.TREND: "green",
    Regime.TOXIC: "red",
}


# --- Raw snapshot ------------------------------------------------------------
@dataclass(slots=True)
class Snapshot:
    """One LOB observation as emitted by the matching engine.

    Prices are stored as *tick offsets from mid* (level 0 = best). Sizes and
    cancel-flow are raw quantities; trade fields summarise the activity that
    occurred since the previous snapshot.
    """
    ts: float
    mid: float
    bid_px: np.ndarray   # (N_LEVELS,) tick offset from mid, >= 0 going deeper
    bid_sz: np.ndarray   # (N_LEVELS,) resting size
    ask_px: np.ndarray   # (N_LEVELS,)
    ask_sz: np.ndarray   # (N_LEVELS,)
    bid_cxl: np.ndarray  # (N_LEVELS,) volume cancelled at level since last snap
    ask_cxl: np.ndarray  # (N_LEVELS,)
    trade_buy: float     # aggressive BUY volume since last snap
    trade_sell: float    # aggressive SELL volume since last snap
    trade_n: int         # number of trades since last snap
    regime: int          # Regime ground truth (eval only)


# Column order written to / read from parquet. Keep stable across versions.
def column_names() -> list[str]:
    cols = ["ts", "mid"]
    for side in ("bid", "ask"):
        cols += [f"{side}_px_{i}" for i in range(N_LEVELS)]
        cols += [f"{side}_sz_{i}" for i in range(N_LEVELS)]
    for side in ("bid", "ask"):
        cols += [f"{side}_cxl_{i}" for i in range(N_LEVELS)]
    cols += ["trade_buy", "trade_sell", "trade_n", "regime"]
    return cols


# Width of one raw snapshot row (== len(column_names())). The native C++ bridge
# POD and this constant must agree for end-to-end zero-copy featurization.
RAW_SNAPSHOT_DOUBLES = 2 + 6 * N_LEVELS + 4  # ts, mid, 60 level cols, 4 trade/eval
COLUMN_INDEX = {name: i for i, name in enumerate(column_names())}


def snapshot_to_row(s: Snapshot) -> dict:
    row = {"ts": s.ts, "mid": s.mid}
    for i in range(N_LEVELS):
        row[f"bid_px_{i}"] = float(s.bid_px[i])
        row[f"bid_sz_{i}"] = float(s.bid_sz[i])
        row[f"ask_px_{i}"] = float(s.ask_px[i])
        row[f"ask_sz_{i}"] = float(s.ask_sz[i])
    for i in range(N_LEVELS):
        row[f"bid_cxl_{i}"] = float(s.bid_cxl[i])
        row[f"ask_cxl_{i}"] = float(s.ask_cxl[i])
    row["trade_buy"] = s.trade_buy
    row["trade_sell"] = s.trade_sell
    row["trade_n"] = s.trade_n
    row["regime"] = int(s.regime)
    return row


def snapshot_to_vector(s: Snapshot) -> np.ndarray:
    """Flatten a Snapshot to the ``(RAW_SNAPSHOT_DOUBLES,)`` raw vector in
    ``column_names()`` order — the exact form pushed into the C++ zero-copy ring
    and read back as a view for ``featurize_from_raw``."""
    row = snapshot_to_row(s)
    return np.array([row[c] for c in column_names()], dtype=np.float64)


# --- Feature layout (model input) -------------------------------------------
# Blocks in order; each entry is (name, width). The model consumes the
# concatenation. Masked-LOB modelling masks whole *level columns* within the
# depth/cancel blocks. The cancel-flow block is kept distinct on purpose so the
# encoder can learn to discount phantom (spoofed) depth (Component D in spec).
FEATURE_BLOCKS: list[tuple[str, int]] = [
    ("bid_px", N_LEVELS),
    ("bid_sz", N_LEVELS),
    ("ask_px", N_LEVELS),
    ("ask_sz", N_LEVELS),
    ("bid_cxl", N_LEVELS),
    ("ask_cxl", N_LEVELS),
    ("trade", 3),   # [signed_flow, log_total_flow, log_count]
]
FEATURE_DIM = sum(w for _, w in FEATURE_BLOCKS)


def block_slices() -> dict[str, slice]:
    out, off = {}, 0
    for name, w in FEATURE_BLOCKS:
        out[name] = slice(off, off + w)
        off += w
    return out


# Indices of the depth/cancel blocks expressed as per-level columns, so masking
# can drop an entire price level (px+sz+cxl) coherently — the model must infer a
# hidden level from its neighbours (true masked-LOB modelling).
def level_column_groups() -> list[list[int]]:
    sl = block_slices()
    groups = []
    for i in range(N_LEVELS):
        groups.append([
            sl["bid_px"].start + i, sl["bid_sz"].start + i,
            sl["ask_px"].start + i, sl["ask_sz"].start + i,
            sl["bid_cxl"].start + i, sl["ask_cxl"].start + i,
        ])
    return groups


def featurize_from_raw(raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Featurize a raw ``(n, RAW_SNAPSHOT_DOUBLES)`` matrix whose columns are in
    ``column_names()`` order — no pandas. This is what runs over the C++ bridge's
    zero-copy view: the input is a view of the shared pages, only the returned
    feature matrix is freshly allocated. The ``regime`` column is excluded (Rule 4).
    """
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != RAW_SNAPSHOT_DOUBLES:
        raise ValueError(
            f"raw must be (n, {RAW_SNAPSHOT_DOUBLES}); got {raw.shape}")
    n = raw.shape[0]
    X = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    sl = block_slices()

    def col(name):
        return raw[:, COLUMN_INDEX[name]].astype(np.float32, copy=False)

    def log_norm(a, scale):
        return np.log1p(np.maximum(a, 0.0)) / scale

    for i in range(N_LEVELS):
        X[:, sl["bid_px"].start + i] = col(f"bid_px_{i}") / TICK_SCALE
        X[:, sl["ask_px"].start + i] = col(f"ask_px_{i}") / TICK_SCALE
        X[:, sl["bid_sz"].start + i] = log_norm(col(f"bid_sz_{i}"), SIZE_SCALE)
        X[:, sl["ask_sz"].start + i] = log_norm(col(f"ask_sz_{i}"), SIZE_SCALE)
        X[:, sl["bid_cxl"].start + i] = log_norm(col(f"bid_cxl_{i}"), CXL_SCALE)
        X[:, sl["ask_cxl"].start + i] = log_norm(col(f"ask_cxl_{i}"), CXL_SCALE)

    buy, sell = col("trade_buy"), col("trade_sell")
    total = buy + sell
    with np.errstate(invalid="ignore", divide="ignore"):  # quiet on edge input
        signed = np.where(total > 0, (buy - sell) / np.maximum(total, 1e-6), 0.0)
    t = sl["trade"]
    X[:, t.start + 0] = signed
    X[:, t.start + 1] = log_norm(total, TRADE_SCALE)
    X[:, t.start + 2] = log_norm(col("trade_n"), 4.0)
    return X, _feature_names()


def featurize(df) -> tuple[np.ndarray, list[str]]:
    """Convert a raw snapshot DataFrame to a normalised feature matrix.

    Thin wrapper over :func:`featurize_from_raw` (the single implementation), so
    the offline parquet path and the live zero-copy bridge path are byte-identical.
    """
    raw = df.loc[:, column_names()].to_numpy(dtype=np.float64)
    return featurize_from_raw(raw)


def feature_weights() -> np.ndarray:
    """Per-feature reconstruction weights.

    The cancel-flow and trade-magnitude blocks carry the regime signal but are a
    small fraction of the dimensions, so the masked-reconstruction loss up-weights
    them; raw price offsets and the *signed* trade direction are down-weighted so
    the embedding encodes micro-structure type rather than which way price ticked.
    """
    sl = block_slices()
    w = np.ones(FEATURE_DIM, dtype=np.float32)
    w[sl["bid_px"]] = 0.5
    w[sl["ask_px"]] = 0.5
    w[sl["bid_cxl"]] = 4.0
    w[sl["ask_cxl"]] = 4.0
    t = sl["trade"].start
    w[t + 0] = 0.5   # signed direction — deliberately minor
    w[t + 1] = 3.0   # log total flow
    w[t + 2] = 3.0   # log trade count
    return w


def mirror_spec() -> tuple[np.ndarray, np.ndarray]:
    """Side-swap permutation for direction-invariance augmentation.

    Returns ``(index, sign)`` such that ``x[:, index] * sign`` is the same book
    reflected across the mid: bid<->ask depth/cancel swapped and the signed trade
    flow negated. An up-trend maps to its mirror down-trend, so an embedding made
    invariant to this transform treats both as the single TREND regime.
    """
    sl = block_slices()
    idx = np.arange(FEATURE_DIM)
    for a, b in (("bid_px", "ask_px"), ("bid_sz", "ask_sz"), ("bid_cxl", "ask_cxl")):
        sa, sb = sl[a], sl[b]
        for i in range(N_LEVELS):
            idx[sa.start + i] = sb.start + i
            idx[sb.start + i] = sa.start + i
    sign = np.ones(FEATURE_DIM, dtype=np.float32)
    sign[sl["trade"].start + 0] = -1.0  # negate signed direction under reflection
    return idx, sign


def _feature_names() -> list[str]:
    names = []
    for name, w in FEATURE_BLOCKS:
        if name == "trade":
            names += ["trade_signed", "trade_logflow", "trade_logn"]
        else:
            names += [f"{name}_{i}" for i in range(w)]
    return names
