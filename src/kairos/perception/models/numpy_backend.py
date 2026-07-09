"""Dependency-free numpy inference backend for the LOB encoder.

Training the self-supervised embedder happens in MLX on Apple-Silicon
(``embedder.train``). *Inference*, however, must run everywhere — inside CI, on
a Linux server, in the reasoning container — so the learned System-1 percept is
available even where MLX is not installed.

This module loads the exact same ``.safetensors`` weights the MLX trainer wrote
and reproduces the encoder forward pass in pure numpy:

    encode(x) = enc3( gelu( enc2( gelu( enc1(x) ) ) ) )

An ``nn.Linear`` computes ``y = x @ W.T + b`` with ``W`` of shape
``(out, in)`` — identical here. ``nn.gelu`` is the *exact* Gaussian error
linear unit (not the tanh approximation), reproduced below with a high-accuracy
``erf`` so cluster assignment matches the MLX path to well within KMeans
tolerance. No third-party dependency is required (``safetensors``/``scipy`` are
used only if already importable).
"""
from __future__ import annotations

import json
import struct

import numpy as np

# MLX/PyTorch safetensors dtype tag -> numpy dtype.
_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def load_safetensors(path: str) -> dict[str, np.ndarray]:
    """Minimal, dependency-free ``.safetensors`` reader.

    Format: 8-byte little-endian header length, then a JSON header mapping each
    tensor name to ``{dtype, shape, data_offsets:[start,end]}``, then the raw
    tensor bytes (offsets relative to the end of the header).
    """
    try:  # prefer the canonical loader when the package happens to be present
        from safetensors.numpy import load_file  # type: ignore

        return dict(load_file(path))
    except Exception:
        pass

    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len).decode("utf-8"))
        blob = fh.read()

    out: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = _DTYPES[meta["dtype"]]
        start, end = meta["data_offsets"]
        arr = np.frombuffer(blob[start:end], dtype=dt)
        out[name] = arr.reshape(meta["shape"]).copy()
    return out


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised erf. Uses scipy when available, else Abramowitz & Stegun
    7.1.26 (max abs error < 1.5e-7 — effectively exact for cluster assignment)."""
    try:
        from scipy.special import erf  # type: ignore

        return erf(x)
    except Exception:
        pass
    sign = np.sign(x)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
              + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-z * z))


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU, matching ``mlx.nn.gelu``: x·½·(1 + erf(x/√2))."""
    return x * 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


class NumpyEncoder:
    """Frozen encoder (enc1→gelu→enc2→gelu→enc3) evaluated in numpy.

    Duck-types the MLX ``LobAutoEncoder`` for the ``embed``/``RegimePredictor``
    call sites: it exposes ``encode(X) -> np.ndarray``.
    """

    def __init__(self, weights: dict[str, np.ndarray]):
        # Cast to float64 for a numerically stable, deterministic forward pass.
        self.w1 = weights["enc1.weight"].astype(np.float64)
        self.b1 = weights["enc1.bias"].astype(np.float64)
        self.w2 = weights["enc2.weight"].astype(np.float64)
        self.b2 = weights["enc2.bias"].astype(np.float64)
        self.w3 = weights["enc3.weight"].astype(np.float64)
        self.b3 = weights["enc3.bias"].astype(np.float64)

    def encode(self, X: np.ndarray) -> np.ndarray:
        h = _gelu(np.asarray(X, dtype=np.float64) @ self.w1.T + self.b1)
        h = _gelu(h @ self.w2.T + self.b2)
        z = h @ self.w3.T + self.b3
        return z.astype(np.float32)

    # Alias so a caller can treat it interchangeably with the MLX module.
    __call__ = encode

    @classmethod
    def load(cls, weights_path: str) -> NumpyEncoder:
        return cls(load_safetensors(weights_path))
