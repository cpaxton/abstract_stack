"""Native-free checkpoint integrity inspection; never imports the simulator."""

import hashlib
import json
from pathlib import Path

from abstract_stack.source_store import resolve_source_reference

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
    source = manifest.get("source_capture")
    if source is not None:
        if source.get("format") != "source_bundle_zip" or type(source.get("portable")) is not bool:
            raise ValueError("Unsupported checkpoint source capture format")
        source_files = {"source_ref.json"}
        if source["portable"]:
            source_files.add("source_bundle.zip")
        if not source_files <= set(manifest["hashes"]):
            raise ValueError("Checkpoint is missing source capture hashes")
    for name, digest in manifest["hashes"].items():
        path = directory / name
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256_file(path) != digest:
            raise ValueError(f"Checkpoint payload hash mismatch: {name}")
    if source is not None:
        reference = json.loads((directory / "source_ref.json").read_text())
        if reference.get("portable") != source["portable"] or (
            source["portable"] and reference.get("archive") != "source_bundle.zip"
        ):
            raise ValueError("Checkpoint source portability contract differs")
        bundle = resolve_source_reference(directory / "source_ref.json", materialize=False)
        if bundle.content_id != source["content_id"]:
            raise ValueError("Checkpoint source identity differs")
    return directory, manifest
