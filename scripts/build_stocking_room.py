"""Export a room MJCF, task manifest, source reference and optional preview."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
from PIL import Image

from abstract_stack.assets import mabel_model_path
from abstract_stack.room_scene import RoomConfig, build_stocking_room
from abstract_stack.source_store import freeze_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stations", type=int, default=2)
    parser.add_argument("--products-per-station", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--product-kind", choices=("mixed", "carton"), default="mixed")
    parser.add_argument("--mabel", action="store_true")
    parser.add_argument("--mabel-root", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.mabel_root and not args.mabel:
        parser.error("--mabel-root requires --mabel")

    root = Path(__file__).resolve().parents[1]
    source = freeze_source(
        root,
        paths=("pyproject.toml", "uv.lock", "src/abstract_stack", "scripts/build_stocking_room.py"),
        materialize=False,
    )
    cfg = RoomConfig(args.stations, args.products_per_station, args.product_kind, args.seed)
    room = build_stocking_room(cfg, mabel=args.mabel, mabel_root=args.mabel_root)
    model = mujoco.MjModel.from_xml_string(room.model_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot_source = None
    if args.mabel:
        path = mabel_model_path(args.mabel_root)
        git_dir = path.parents[3] / ".git"
        revision = None
        if git_dir.exists():
            revision = subprocess.check_output(
                ["git", "-C", str(git_dir.parent), "rev-parse", "HEAD"], text=True
            ).strip()
        robot_source = {
            "path": str(path),
            "revision": revision,
            "model_xml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=False)
    source.write_reference(directory / "source_reference.json")
    (directory / "room.xml").write_text(room.model_xml, encoding="utf8")
    manifest = {
        **room.manifest(),
        "task_evidence": "fixture_only",
        "mujoco": mujoco.__version__,
        "timestep_s": model.opt.timestep,
        "initial_qpos": data.qpos.tolist(),
        "robot_source": robot_source,
        "source_bundle_id": source.content_id,
        "model_xml_sha256": hashlib.sha256(room.model_xml.encode()).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    if args.render:
        view = mujoco.MjvOption()
        view.geomgroup[5] = 0
        view.sitegroup[4] = 1
        with mujoco.Renderer(model, width=640, height=480) as renderer:
            renderer.update_scene(data, camera="room_overview", scene_option=view)
            Image.fromarray(renderer.render()).save(directory / "room_overview.png")


if __name__ == "__main__":
    main()
