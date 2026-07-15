"""kairos — the unified dual-process trading brain CLI.

    kairos version
    kairos loop      [--scenario toxic|calm|range] [--mode deterministic|llm] [--steps N]
    kairos perceive  <perception subcommand ...>   # System-1 (LOB-Core): gen/train/cluster/backtest/web/...
    kairos reason    <TICKER> <YYYY-MM-DD> [--json]  # System-2 (TradingAgents) — needs [reasoning]
    kairos web       [--live] [--port N]            # live regime dashboard
    kairos soul-check                               # the unified Constitution enforcer
    kairos reproduce                                # end-to-end reproducibility gate

`loop` is the flagship: it runs perceive→reason→act→reflect and prints the
reflection. The System-1 and bridge paths need no API keys and no MLX; `reason`
and `--mode llm` require the ``[reasoning]`` extra.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__

_BANNER = r"""
 ██   ██  █████  ██ ██████   ██████  ███████
 ██  ██  ██   ██ ██ ██   ██ ██    ██ ██
 █████   ███████ ██ ██████  ██    ██ ███████
 ██  ██  ██   ██ ██ ██   ██ ██    ██      ██
 ██   ██ ██   ██ ██ ██   ██  ██████  ███████   the AURA dual-process trading brain
     System-1 perception (LOB-Core)  ×  System-2 reasoning (TradingAgents)
"""


def _cmd_loop(args) -> int:
    from kairos.loop import LoopConfig, run_cognitive_loop

    if not 0.0 < args.decision_fraction < 1.0:  # a fraction of the session, exclusive of both ends
        print(f"--decision-fraction must be in (0, 1), got {args.decision_fraction}.\n"
              "  It is the point through the session where the stance is set.",
              file=sys.stderr)
        return 2
    cfg = LoopConfig(symbol=args.symbol, scenario=args.scenario, n_steps=args.steps,
                     seed=args.seed, mode=args.mode, decision_fraction=args.decision_fraction)
    if args.learned:
        try:
            cfg.regime_backend = _load_learned_backend()
        except FileNotFoundError as exc:  # encoder artifacts are gitignored (Apple-Silicon + MLX only)
            print(f"--learned needs the trained System-1 encoder, but an artifact is missing: {exc}\n"
                  "  Produce it with `kairos perceive --mode synthetic` (gen→train→cluster).\n"
                  "  Training needs Apple Silicon + MLX: `pip install 'kairos[mlx]'`.\n"
                  "  Or drop --learned to run the loop on the deterministic backend.",
                  file=sys.stderr)
            return 2
    try:
        result = run_cognitive_loop(cfg)
    except ValueError as exc:  # e.g. --steps too small to leave a forward window past perception_window
        print(f"the loop has no forward window to execute over: {exc}\n"
              f"  Raise --steps (needs more than perception_window={cfg.perception_window} rows past the "
              "decision point), or lower --decision-fraction.",
              file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(_BANNER)
        print(result.reflection)
    return 0


def _load_learned_backend():
    """Load the trained System-1 regime backend (numpy or MLX inference)."""
    from kairos.bridge import LearnedRegime
    from kairos.perception.regime.predict import RegimePredictor

    return LearnedRegime(RegimePredictor.load())


def _delegate_perceive(rest: list[str]) -> int:
    from kairos.perception.cli import main as perception_main

    return int(perception_main(rest) or 0)


def _valid_date(value: str) -> str:
    """argparse ``type=`` for the reason date: YYYY-MM-DD, not in the future.

    Mirrors the interactive path (reasoning_cli) so a bad date fails fast with a
    clean CLI error (exit 2) before any LLM/API-key setup, rather than surfacing
    an unrelated 'API key not set' error deep in propagation.
    """
    from datetime import date, datetime

    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}: use YYYY-MM-DD") from None
    if parsed > date.today():
        raise argparse.ArgumentTypeError(f"date {value!r} is in the future")
    return value


def _cmd_reason(args) -> int:
    try:
        from kairos.reasoning.default_config import DEFAULT_CONFIG
        from kairos.reasoning.graph.trading_graph import TradingAgentsGraph
    except ImportError as exc:  # pragma: no cover - optional extra
        print(f"System-2 reasoning needs the [reasoning] extra: {exc}\n"
              "  pip install 'kairos[reasoning]'", file=sys.stderr)
        return 2
    ta = TradingAgentsGraph(debug=args.debug, config=DEFAULT_CONFIG.copy())
    _, decision = ta.propagate(args.ticker, args.date, asset_type=args.asset_type)
    if args.json:
        print(json.dumps({"ticker": args.ticker, "date": args.date,
                          "asset_type": args.asset_type, "decision": decision}, default=str))
    else:
        print(decision)
    return 0


def _cmd_web(args) -> int:
    argv = ["web"]
    if args.live:
        argv.append("--live")
    if args.port:
        argv += ["--port", str(args.port)]
    from kairos.perception.cli import main as perception_main

    return int(perception_main(argv) or 0)


def _run_source_script(name: str, command: str) -> int:
    """Run a scripts/ audit tool, or explain that it needs a source checkout.

    ``scripts/`` (and the src/tests/ trees these tools scan) is not shipped in
    the installed wheel, so from a `pip install`ed kairos the script is absent;
    detect that and print a clear message with exit 2 rather than letting
    subprocess emit a raw file-not-found.
    """
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / name
    if not script.is_file():
        print(f"`kairos {command}` is only available from a source checkout: {script} is missing.\n"
              "  It audits the on-disk src/tests/scripts trees, which are not shipped in the "
              "installed wheel.\n"
              "  Clone the repo (github.com/cleoanka/kairos) and run it from there.",
              file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, str(script)])


def _cmd_soul(args) -> int:
    return _run_source_script("soul_check.py", "soul-check")


def _cmd_reproduce(args) -> int:
    return _run_source_script("reproduce.py", "reproduce")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kairos", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"kairos {__version__}")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("version", help="print the version").set_defaults(func=lambda a: print(__version__) or 0)

    lp = sub.add_parser("loop", help="run the perceive→reason→act→reflect cognitive loop")
    lp.add_argument("--symbol", default="BTCUSDT")
    lp.add_argument("--scenario", default="toxic", choices=["toxic", "calm", "range"])
    lp.add_argument("--mode", default="deterministic", choices=["deterministic", "llm"])
    lp.add_argument("--steps", type=int, default=4000)
    lp.add_argument("--seed", type=int, default=7)
    lp.add_argument("--decision-fraction", dest="decision_fraction", type=float, default=0.5)
    lp.add_argument("--learned", action="store_true", help="use the trained System-1 regime backend")
    lp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    lp.set_defaults(func=_cmd_loop)

    # `perceive` is intercepted in main() before argparse so its own options
    # (e.g. --mode synthetic) pass straight through to the perception CLI; this
    # registration exists only so it appears in `kairos --help`.
    sub.add_parser("perceive", add_help=False,
                   help="System-1 (LOB-Core) subcommands (gen/train/cluster/backtest/...)")

    re = sub.add_parser("reason", help="System-2 (TradingAgents) decision for a ticker/date")
    re.add_argument("ticker")
    re.add_argument("date", type=_valid_date, help="analysis date YYYY-MM-DD (not in the future)")
    re.add_argument("--asset-type", default="stock", choices=["stock", "crypto"])
    re.add_argument("--debug", action="store_true")
    re.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    re.set_defaults(func=_cmd_reason)

    wb = sub.add_parser("web", help="serve the live regime dashboard")
    wb.add_argument("--live", action="store_true")
    wb.add_argument("--port", type=int, default=0)
    wb.set_defaults(func=_cmd_web)

    sub.add_parser("soul-check", help="run the unified Constitution enforcer").set_defaults(func=_cmd_soul)
    sub.add_parser("reproduce", help="end-to-end reproducibility gate").set_defaults(func=_cmd_reproduce)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `perceive` forwards ALL following args to the perception CLI verbatim, so
    # argparse never tries to interpret the perception CLI's own flags.
    if argv and argv[0] == "perceive":
        return _delegate_perceive(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        print(_BANNER)
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
