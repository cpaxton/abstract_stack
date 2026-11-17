"""Native-free checkpoint integrity inspection; never imports the simulator."""

import hashlib
import json
from pathlib import Path

SCHEMA = 1


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def inspect_checkpoint(directory):
    directory = Path(directory)
    if not (directory / "manifest.json").exists() and (directory / "checkpoint").is_dir():
        directory /= "checkpoint"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Checkpoint is incomplete or has an unsupported schema")
    required = {
        "model.mjb",
        "state.npz",
        "scene.json",
        "controller.json",
        "context.json",
        "random.json",
        "witness.npz",
    }
    if not required <= set(manifest.get("hashes", {})):
        raise ValueError("Checkpoint is missing required payload hashes")
    for name, digest in manifest["hashes"].items():
        path = directory / name
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256_file(path) != digest:
            raise ValueError(f"Checkpoint payload hash mismatch: {name}")
    return directory, manifest
