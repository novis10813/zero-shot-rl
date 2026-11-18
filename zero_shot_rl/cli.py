"""Lightweight subcommand dispatcher for zero_shot_rl.{train,finetune,eval}."""

import runpy
import sys


def dispatch(package, commands, aliases=None):
    aliases = aliases or {}
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(f"Usage: python -m {package} <command> [args...]\n\nAvailable commands:")
        width = max(len(k) for k in commands)
        for name, (mod, desc) in commands.items():
            print(f"  {name:{width}s}  {desc} ({mod})")
        if aliases:
            print("\nAliases: " + ", ".join(f"{k} -> {v}" for k, v in aliases.items()))
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 2)

    cmd = sys.argv[1]
    cmd = aliases.get(cmd, cmd)
    if cmd not in commands:
        known = ", ".join(sorted([*commands, *aliases]))
        print(f"Unknown command {sys.argv[1]!r}. Choose from: {known}", file=sys.stderr)
        sys.exit(2)

    module_name, _ = commands[cmd]
    sys.argv = [module_name, *sys.argv[2:]]
    runpy.run_module(module_name, run_name="__main__")
