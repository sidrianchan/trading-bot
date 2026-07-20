"""CLI handlers for `python main.py evolve <subcommand>`.

main.py stays a thin dispatcher; all evolution logic lives in evolve/.
"""
from __future__ import annotations

from pathlib import Path

USAGE = """\
Usage: python main.py evolve <subcommand>

  status              show registry: strategies, active versions, params
  seed                seed v1 registry records for the two production strategies
  run [--dry-run]     run the weekly evolution cycle (dry-run: print digest, no LLM)
  shadow-step         daily: mark shadow books, check envelopes, monitor live DD
  approve <id> [--note "..."]   approve a pending promotion proposal
  reject  <id> [--note "..."]   reject a pending promotion proposal
  ramp    <strategy_id>         lift a promoted strategy to 100% capital
  demote  <strategy_id>         revert ACTIVE to the previous version
"""


def _flag_value(argv: list[str], flag: str) -> str:
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return ""


def run_evolve_command(config: dict, argv: list[str]) -> None:
    args = [a for a in argv if not a.startswith("--")]
    note = _flag_value(argv, "--note")
    if note and note in args:
        args.remove(note)
    sub = args[0] if args else "status"
    target = args[1] if len(args) > 1 else None

    if sub == "status":
        _cmd_status(config)
    elif sub == "seed":
        _cmd_seed(config)
    elif sub == "run":
        from evolve.cycle import run_cycle

        run_cycle(config, dry_run="--dry-run" in argv)
    elif sub == "shadow-step":
        from evolve.shadow import shadow_step

        shadow_step(config)
        from evolve.monitor_live import check_live_envelopes

        check_live_envelopes(config)
    elif sub in {"approve", "reject"} and target:
        from evolve.promote import approve_proposal, reject_proposal

        if sub == "approve":
            record = approve_proposal(target, note=note)
            print(f"Approved {target}: {record['strategy_id']} -> v{record['version']} "
                  f"at {record['capital_fraction']:.0%} capital")
        else:
            reject_proposal(target, note=note)
            print(f"Rejected {target}")
    elif sub == "ramp" and target:
        from evolve.promote import ramp_strategy

        ramp_strategy(target)
    elif sub == "demote" and target:
        from evolve.promote import demote_strategy

        record = demote_strategy(target, actor="human", reason=note or "manual demotion")
        print(f"Demoted {target}; ACTIVE now v{record.get('version')}")
    else:
        print(USAGE)


def _cmd_seed(config: dict) -> None:
    from evolve import registry

    created = registry.seed_defaults(config)
    if created:
        print("Seeded registry records: " + ", ".join(created))
    else:
        print("Registry already seeded; nothing to do.")


def _cmd_status(config: dict) -> None:
    from evolve import registry
    from evolve.guardrails import HARD, evolution_allowed

    strategies = registry.list_strategies()
    print("\nSTRATEGY REGISTRY")
    print("=" * 72)
    if not strategies:
        print("  (empty — run `python main.py evolve seed` to register the live strategies)")
    for rec in strategies:
        print(f"\n  {rec['strategy_id']}  v{rec['version']}  [{rec['status']}]"
              f"  ({rec['n_versions']} version(s) on record)")
        print(f"    family           : {rec['family']}")
        print(f"    capital fraction : {rec['capital_fraction']:.0%}")
        for name, value in sorted(rec["params"].items()):
            print(f"    {name:<28}: {value}")
        expected = (rec.get("provenance") or {}).get("expected") or {}
        if expected:
            parts = [f"{k} {v:+.1%}" if isinstance(v, float) and k != "sharpe"
                     else f"{k} {v}" for k, v in expected.items() if v is not None]
            print(f"    expected (backtest)          : {', '.join(parts)}")

    allowed, reason = evolution_allowed()
    print("\nEVOLUTION")
    print("=" * 72)
    print(f"  evolution allowed : {'yes' if allowed else f'NO — {reason}'}")
    print(f"  hard limits       : max {HARD.max_candidates_per_cycle} candidates/cycle, "
          f"min shadow {HARD.min_shadow_days}d, "
          f"promotion ramp {HARD.new_strategy_capital_fraction:.0%} for {HARD.ramp_days}d")
    db = Path("logs") / "evolve" / "evolve.db"
    if db.exists():
        print(f"  proposal DB       : {db}")
    else:
        print("  proposal DB       : not initialized (created by the first `evolve run`)")
    print()
