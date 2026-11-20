#!/usr/bin/env python3
"""Standalone MuJoCo named-view lifetime probe; no robot assets or controller.

Run each mode in a fresh process. A native crash is a failed diagnostic, never
an accepted simulation result. Retaining wrappers is a comparison, not proof of
the underlying library defect or a universally safe workaround.
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("transient", "retained", "indexed"), required=True)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--bodies", type=int, default=20)
    parser.add_argument("--copy-every", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or not 1 <= args.bodies <= 100 or args.copy_every < 0:
        parser.error("Require positive iterations, 1–100 bodies and nonnegative copy interval")
    args.output.mkdir(parents=True, exist_ok=False)
    xml = (
        "<mujoco><worldbody>"
        + "".join(
            f'<body name="body_{i}"><joint name="joint_{i}"/>'
            f'<geom size=".01" pos="{i * 0.05} 0 0"/></body>'
            for i in range(args.bodies)
        )
        + "</worldbody></mujoco>"
    )
    (args.output / "model.xml").write_text(xml + "\n")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = [model.body(i).name for i in range(1, model.nbody)]
    retained = [model.body(name) for name in names] if args.mode == "retained" else []
    manifest = {
        "kind": "primitive_named_view_diagnostic_no_physical_task_credit",
        "mode": args.mode,
        "iterations": args.iterations,
        "bodies": args.bodies,
        "copy_every": args.copy_every,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "physics_steps": 0,
        "retained_model_views": len(retained),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    with (args.output / "progress.jsonl").open("w", buffering=1) as progress:
        for iteration in range(args.iterations):
            for index, name in enumerate(names):
                if args.mode == "indexed":
                    point = data.xpos[index + 1].copy()
                    rotation = data.xmat[index + 1].reshape(3, 3).copy()
                else:
                    assert model.body(name).id == index + 1
                    point = data.body(name).xpos.copy()
                    rotation = data.body(name).xmat.reshape(3, 3).copy()
                assert np.isfinite(point).all() and rotation[2, 2] > 0
            if args.copy_every and iteration % args.copy_every == 0:
                scratch = mujoco.MjData(model)
                mujoco.mj_copyData(scratch, model, data)
                mujoco.mj_forward(model, scratch)
            if iteration % 1000 == 0:
                gc.collect()
                progress.write(json.dumps({"iteration": iteration}) + "\n")
    result = dict(manifest, completed=True, wall_seconds=time.monotonic() - started)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
