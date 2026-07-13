# Security Policy

Kairos is a research system, but it touches two sensitive surfaces: **LLM API
credentials** and (optionally) **live market connectivity**. Please treat both
with care.

## Reporting a vulnerability

Do **not** open a public issue for a security problem. Instead, use GitHub's
private advisory flow:

> **Security → Report a vulnerability** on
> <https://github.com/cleoanka/kairos/security/advisories/new>

Please include a description, reproduction steps, affected version/commit, and
the impact. We aim to acknowledge within a few days and to coordinate a fix and
disclosure timeline with you.

## Supported versions

Kairos is pre-1.x in spirit (current release `1.0.0`); only the latest `main`
receives security fixes.

## Scope and hardening notes

- **Secrets.** LLM keys are read from the environment / `.env` (see
  `.env.example`) and are **never** committed. `.env` is git-ignored. The
  reasoning tests run with *placeholder* keys and make no live API calls.
- **Live trading is refused by default.** Real-money order routing is
  deliberately gated behind explicit operator confirmation and credentials; the
  default paths run on synthetic or replayed markets only.
- **No network in the causal core.** The System-1 execution path forbids
  REST/HTTP (Constitution Rule 3, enforced by `scripts/soul_check.py`); the only
  outbound network is the opt-in public market-data WebSocket dashboard and the
  opt-in reasoning LLM calls.
- **Look-ahead is a safety property here.** The Causal Perception Bus guarantees
  System 2 can only read percepts at or before its cutoff. If you find a way to
  make a dated query reach a future percept, that is an in-scope report.

## Out of scope

This software is for research and carries **no financial-advice warranty**.
Losses from any use — including live routing you enable yourself — are your
responsibility (see the disclaimer in the [README](README.md) and [`LICENSE`](LICENSE)).
