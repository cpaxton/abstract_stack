"""Export a scene, its initial state and provenance, and optional rendered previews."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import mujoco

from abstract_stack.assets import MABEL_REVISION, mabel_model_path
from abstract_stack.provenance import policy_provenance


def build_scene_bundle(env, directory: str | Path, *, render: bool = False) -> Path:
    """Reset the environment, export its initial scene, then settle it for inspection."""
    env.reset()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    xml = env.export(directory / "scene.xml")
    versions = {"mujoco": mujoco.__version__}
    for name in ("abstract-stack", "emet", "numpy"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed as a distribution"
    source = None
    if env.robot:
        model_path = mabel_model_path(env.mabel_root)
        result = subprocess.run(
            ["git", "-C", str(model_path.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        source = {
            "model_path": str(model_path),
            "revision": result.stdout.strip() if result.returncode == 0 else None,
            "default_revision": MABEL_REVISION,
        }
    initial_poses = env.observe()["product_poses"]
    manifest = {
        "schema_version": 1,
        "policy_source": policy_provenance(),
        "seed": env.info()["seed"],
        "robot": "mabel" if env.robot else "none",
        "robot_source": source,
        "versions": versions,
        "config": asdict(env.config),
        "model": {"nq": env.model.nq, "nv": env.model.nv, "nu": env.model.nu},
        "coordinates": "metres, radians, MuJoCo world frame; shelf opens toward +X",
        "initial_qpos": env.data.qpos.tolist(),
        "initial_ctrl": env.data.ctrl.tolist(),
        "products": [
            {**asdict(product), "initial_pose": initial_poses[i].tolist()}
            for i, product in enumerate(env.products)
        ],
        "cameras": [env.model.camera(i).name for i in range(env.model.ncam)],
        "xml": xml.name,
    }
    if env.config.environment == "small_house":
        from abstract_stack.house import house_layout

        manifest["environment"] = house_layout()
    else:
        manifest["environment"] = {"kind": "studio"}
    # Settle under normal dynamics to make previews useful for checking support
    # and stability. The XML and initial state above remain the episode start.
    for _ in range(50):
        _, _, terminated, truncated, _ = env.step()
        if terminated or truncated:
            break
    manifest["settled_metrics"] = env.info()
    if render:
        from PIL import Image

        cameras = ["overview", "shelf_front", "supply_overhead"]
        if env.config.environment == "small_house":
            cameras.extend(["house_overhead", "kitchen", "living_room", "bedroom", "bathroom"])
        for camera in cameras:
            Image.fromarray(env.render(camera)).save(directory / f"{camera}.png")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory
