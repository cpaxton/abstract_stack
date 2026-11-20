"""Load editable JSON scene configurations and bundled starting layouts."""

from __future__ import annotations

import json
from dataclasses import fields
from importlib.resources import files
from pathlib import Path

from abstract_stack.scene import ShelfConfig, products_for

SCENES = ("stocking", "empty", "tight", "house", "house_tight", "pick_place")


def load_scene_config(path: str | Path | None = None, *, scene: str = "stocking") -> ShelfConfig:
    if scene not in SCENES:
        raise ValueError(f"Unknown scene {scene!r}; choose from {', '.join(SCENES)}")
    resource = (
        Path(path)
        if path is not None
        else files("abstract_stack").joinpath("configs").joinpath(f"{scene}.json")
    )
    values = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Scene configuration must be a JSON object")
    unknown = values.keys() - {field.name for field in fields(ShelfConfig)}
    if unknown:
        raise ValueError(f"Unknown scene configuration fields: {', '.join(sorted(unknown))}")
    for key in ("levels", "table_size", "robot_xy", "carton_size"):
        if key in values:
            if not isinstance(values[key], list):
                raise ValueError(f"{key} must be a JSON array")
            values[key] = tuple(values[key])
    for key in ("supply_order", "carton_masses_kg"):
        if values.get(key) is not None:
            if not isinstance(values[key], list):
                raise ValueError(f"{key} must be a JSON array")
            values[key] = tuple(values[key])
    if "initial_joint_positions" in values:
        try:
            values["initial_joint_positions"] = tuple(
                tuple(pair) for pair in values["initial_joint_positions"]
            )
        except TypeError as error:
            raise ValueError(
                "initial_joint_positions must be an array of name/value pairs"
            ) from error
    try:
        config = ShelfConfig(**values)
        products_for(config)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid scene configuration: {error}") from error
    return config
