#!/usr/bin/env python3
"""Export a compact checkpoint at any physics step in a recorded skill attempt."""

import argparse
import json
from pathlib import Path

from abstract_stack.skill_replay import capture_skill_cut, resume_skill_cut
from abstract_stack.source_store import default_source_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--attempt", type=Path)
    mode.add_argument("--resume", type=Path, help="Replay the suffix from a saved cut")
    parser.add_argument("--physics-step", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.resume:
        print(json.dumps(resume_skill_cut(args.resume)))
        return
    if args.physics_step is None or args.output is None:
        parser.error("Capturing an attempt requires --physics-step and --output")
    root = Path(__file__).resolve().parents[1]
    cache = default_source_store(root).parent / "cache/checkpoint_models"
    print(
        json.dumps(
            capture_skill_cut(args.attempt, args.physics_step, args.output, model_cache=cache)
        )
    )


if __name__ == "__main__":
    main()
