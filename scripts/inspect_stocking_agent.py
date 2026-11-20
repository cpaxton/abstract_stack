#!/usr/bin/env python3
"""Show a live or completed run's stocking plan, progress and recovery branches."""

import argparse
import json
from pathlib import Path

from abstract_stack.agent_history import inspect_history, read_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory or agent-history.jsonl file")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable status")
    parser.add_argument("--events", action="store_true", help="Emit complete journal records")
    args = parser.parse_args()
    if args.events:
        for row in read_history(args.run):
            print(json.dumps(row))
        return
    status = inspect_history(args.run)
    if args.json:
        print(json.dumps(status, indent=2))
        return
    print(f"{status['run_kind']} | {status['status']} | record {status['last_sequence']}")
    cursor = status["current"]
    if cursor:
        fields = ("carton_step", "product", "skill_step", "skill", "phase", "branch", "attempt")
        print(" | ".join(f"{key}={cursor[key]}" for key in fields if cursor.get(key) is not None))
        if cursor.get("selecting"):
            print("Selecting a carton")
    for name, evidence in status["products"].items():
        stages = [key for key, verified in evidence["gates"].items() if verified]
        used = ", ".join(evidence["skills_used"]) or "unreached"
        print(f"{name} | used {used} | verified gates {', '.join(stages) or 'none'}")
        if evidence["skills_verified"]:
            print(f"  verified skills {', '.join(evidence['skills_verified'])}")
        if evidence["unreached_steps"]:
            print(f"  unreached steps {', '.join(evidence['unreached_steps'])}")
        for attempt in evidence["attempts"]:
            if attempt["event"] != "attempt_finished":
                continue
            print(
                f"  {attempt['branch']} #{attempt['attempt']} {attempt['action']} "
                f"{attempt['outcome']} {attempt.get('failure') or ''}".rstrip()
            )
        latest = status["latest_checkpoints"].get(name)
        if latest:
            print(f"  latest checkpoint {latest['checkpoint']}")
    if status["terminal"] and status["terminal"].get("failure"):
        print(status["terminal"]["failure"])


if __name__ == "__main__":
    main()
