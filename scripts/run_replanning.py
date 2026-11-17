#!/usr/bin/env python3
"""Development entry point; ten-carton native task, or read-only order probe."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from abstract_stack.control_profiles import resolve_control_settings
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.interactions import ScriptConfig
from abstract_stack.packing import dense_shelf_config
from abstract_stack.receding_stocking import RecedingStockingScript
from abstract_stack.replanning import SearchConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objects", type=int, default=10)
    parser.add_argument("--total-free-space", type=float, default=0.01)
    parser.add_argument("--mabel-root", type=Path)
    parser.add_argument("--search-seconds", type=float, default=5.0)
    parser.add_argument("--alignment-seconds", type=float, default=6.0)
    parser.add_argument("--control-config", type=Path)
    parser.add_argument("--guarded-approach", action="store_true")
    parser.add_argument("--base-reposition", action="store_true")
    parser.add_argument("--finger-load-compensation", action="store_true")
    parser.add_argument("--bounded-contact-ik", action="store_true")
    parser.add_argument("--square-backstop", action="store_true")
    parser.add_argument("--contact-recovery", action="store_true")
    parser.add_argument("--payload-load-compensation", action="store_true")
    parser.add_argument("--split-insertion", action="store_true")
    parser.add_argument("--video", action="store_true", help="Record synchronized native views")
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument(
        "--grasp-wrist-posture",
        type=float,
        help="Nominal grasp/carry wrist planning posture; preserves native limits",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--admit-loaded-route", action="store_true")
    parser.add_argument(
        "--table-front", action="store_true", help="Release on the existing flush table"
    )
    parser.add_argument(
        "--loaded-route", action="store_true", help="Try low Cartesian loaded corridor"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = dense_shelf_config(
        args.seed, product_count=args.objects, total_free_space=args.total_free_space
    )
    (args.output_dir / "scene.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    search = SearchConfig(wall_seconds=args.search_seconds)
    (args.output_dir / "search_config.json").write_text(json.dumps(asdict(search), indent=2) + "\n")
    env = ShelfStockingEnv(config, mabel_root=args.mabel_root, seed=args.seed)
    try:
        controls = resolve_control_settings("selected", args.control_config)
        controls.setdefault("alignment_timeout_seconds", args.alignment_seconds)
        controls["guarded_approach"] = args.guarded_approach
        solver = RecedingStockingScript(
            env,
            args.output_dir,
            ScriptConfig(**controls),
            force_profile="selected",
            search_config=search,
            loaded_route=args.loaded_route,
            route_admission=args.admit_loaded_route,
            table_front=args.table_front,
            base_reposition=args.base_reposition,
            finger_load_compensation=args.finger_load_compensation,
            bounded_contact_ik=args.bounded_contact_ik,
            square_backstop=args.square_backstop,
            contact_recovery=args.contact_recovery,
            grasp_wrist_posture=args.grasp_wrist_posture,
            payload_load_compensation=args.payload_load_compensation,
            split_insertion=args.split_insertion,
            video_options=(
                dict(fps=args.video_fps, width=args.video_width, height=args.video_height)
                if args.video
                else None
            ),
        )
        if args.plan_only:
            # The regular episode runner owns settling and the StageObserver.
            # This probe performs no MuJoCo step and earns no manipulation credit.
            import numpy as np

            from abstract_stack.provenance import policy_provenance

            before = env.data.qpos.copy(), env.data.ctrl.copy(), float(env.data.time)
            failure = None
            try:
                solver._select_grasp(list(env.products))
            except RuntimeError as error:
                failure = str(error)
            unchanged = (
                np.array_equal(before[0], env.data.qpos)
                and np.array_equal(before[1], env.data.ctrl)
                and before[2] == env.data.time
            )
            result = {
                "kind": "reset_state_order_probe_no_physics_or_task_credit",
                "failure": failure,
                "live_state_unchanged": unchanged,
                "source": policy_provenance(),
                "events": solver.events,
                "plans": solver.order_plans,
            }
            (args.output_dir / "plan_probe.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k not in ("source", "events")}))
            return 0 if unchanged and not failure else 2
        result = solver.run()
        print(
            json.dumps({k: result.get(k) for k in ("task_success", "physical_success", "failure")})
        )
        return 0 if result.get("task_success", result["physical_success"]) else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
