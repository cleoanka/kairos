# Contributing to Kairos

Thanks for your interest. Kairos is a source-available research system with a
deliberately strict spine — two systems fused under a look-ahead guarantee that
holds **by construction**. Contributions are welcome as long as they keep that
spine intact and every claim they add is verified true.

> **License note.** Kairos is proprietary (© 2026 cleoanka; see [`LICENSE`](LICENSE)),
> and bundles TradingAgents (© Tauric Research, Apache-2.0) under
> `kairos.reasoning` (see [`NOTICE`](NOTICE)). By contributing you agree your
> changes are licensed under the same umbrella.

## Dev setup

```bash
git clone https://github.com/cleoanka/kairos && cd kairos
make install          # fresh .venv + core + [dev,viz] — portable, no keys, no GPU
make gate             # soul_check + core tests + loop smoke  → all green
```

Optional extras (only if you touch those layers):

```bash
make install-all      # + [reasoning] (LLM debate) + [native] (C++ ring) + [mlx] (train)
```

The whole core loop runs on synthetic markets with no API keys, no GPU, and no
market-data account.

## The verification gate (run before every PR)

```bash
make soul             # the scoped Constitution — must exit 0
make lint             # ruff over Kairos-original code (see scope below)
make test-core        # bridge + loop + perception  (no LLM/MLX/native needed)
make reproduce        # the honest end-to-end reproducibility gate
```

`make test-core` also carries the **scoped mypy gate**: `tests/loop/test_mypy_gate.py`
runs `python -m mypy` (reading `[tool.mypy]` in `pyproject.toml`) and fails the
suite on any type error. Mirroring how ruff only lints Kairos-original code, the
gate type-checks exactly the neuro-symbolic core — `src/kairos/bridge`,
`src/kairos/loop`, and `src/kairos/cli.py` — and leaves the vendored
`kairos.reasoning` tree (upstream typing debt) out on purpose. It is pinned to
`python_version = "3.13"`, so it type-checks against 3.13 even when the `core`
job runs on 3.11 / 3.12.

If you touched `kairos.reasoning`, also run `make test-all` with the
`[reasoning]` extra installed (placeholder API keys are enough — no test makes a
live call).

CI runs the same steps on every push and PR (`.github/workflows/ci.yml`): the
`core` job on Python 3.11 and 3.12 runs soul_check + lint + core tests (mypy gate
included) + `reproduce.py`; the `reasoning` job runs the TradingAgents graph and
microstructure-integration tests.

## The Constitution (`scripts/soul_check.py`) — non-negotiable

Kairos fuses two systems with **deliberately different souls**, so the
constitution is **scoped**. `soul_check` must stay green; a violation fails CI.

- **System 1 (engine + bridge)** — no classic price-lagged TA (RSI/MACD/EMA…),
  no `memcpy`/`memmove` on the LOB hot path, no REST/HTTP in the execution path,
  no supervised direction/regime label as a training target.
- **System 2 (`kairos.reasoning`)** — *exempt* from the no-TA rule; an LLM may
  legitimately reason about RSI/MACD as fallible evidence.
- **Rule 5 (Causality)** — the reasoning-facing bridge (microstructure tools +
  analyst) may read perception **only** through the causal accessors
  `as_of` / `window_before` / `aggregate_before` — never `.latest` / `._percepts`.
  This is the exact look-ahead hole Kairos exists to close; touching a
  non-causal accessor there is a hard violation.

If you have a legitimate reason to trip Rule 1 on a genuinely cold path, waive it
locally with a `soul:allow-memcpy` comment — never by weakening the checker.

## Lint scope

`make lint` (and CI) lint **Kairos-original code only**:

```
src/kairos/bridge  src/kairos/loop  src/kairos/cli.py  src/kairos/__init__.py
scripts/soul_check.py  scripts/reproduce.py  tests/bridge  tests/loop
```

The vendored System-1/System-2 subtrees (`kairos.perception`,
`kairos.reasoning`) intentionally follow their **upstream** style and are not
reformatted here. Ruff config lives in `pyproject.toml` (line length 100,
target `py310`).

## Claim discipline

Every number you write in a doc, badge, or comment (test counts, benchmarks,
ARI, tolerances) must be **re-verified against reality** first — re-run the
suite, read the code. Prefer the live CI status badge over a hardcoded count so
it cannot drift. Never claim profit in a regime `reproduce.py` does not assert.

## Pull requests

- Keep commits logical and small; explain *why*, not just *what*.
- Include the verification-gate output (or the CI run) in the PR description.
- Don't add heavy dependencies to the core layer — put anything non-portable
  behind an extra (`[reasoning]` / `[mlx]` / `[native]` / `[viz]`).
