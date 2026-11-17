#!/usr/bin/env python3
"""Audit old diagnostic bundles or export causal LfD episodes locally."""

import argparse
import json
from pathlib import Path

from abstract_stack.lfd.legacy import audit_legacy_bundle
from abstract_stack.lfd.lerobot import export_lerobot, verify_lerobot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-legacy")
    audit.add_argument("episode", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("lerobot")
    export.add_argument("episodes", nargs="+", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--include-failed", action="store_true")
    verify = commands.add_parser("verify-lerobot")
    verify.add_argument("dataset", type=Path)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit-legacy":
        args.output.mkdir(parents=True, exist_ok=False)
        result = audit_legacy_bundle(args.episode)
        (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 2 if not result["training_ready"] else 0
    if args.command == "verify-lerobot":
        args.output.mkdir(parents=True, exist_ok=False)
        result = verify_lerobot(args.dataset)
        (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    result = export_lerobot(args.episodes, args.output, include_failed=args.include_failed)
    print(
        json.dumps(dict(episodes=len(result["episodes"]), output=str(args.output), uploaded=False))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
