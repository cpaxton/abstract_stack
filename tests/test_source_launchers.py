"""Source/reference launcher contracts; no native worker is started by these tests."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from abstract_stack import checkpoint_manifest, source_store


@pytest.mark.parametrize(
    "name",
    [
        "continue_stocking",
        "probe_compliant_lookahead",
        "render_checkpoint_chain",
        "evaluate_squaring_focus",
    ],
)
def test_launchers_pin_references_and_share_store_when_invoked_outside_repo(
    tmp_path, monkeypatch, name
):
    script = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    module = runpy.run_path(str(script))
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    checkpoint = working / "fixture"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text('{"fixture": true}\n')
    segment = working / "segment"
    segment.mkdir()
    for filename, contents in (
        ("continuation.npz", b"recorded controls fixture"),
        ("summary.json", b"{}"),
        ("manifest.json", b"{}"),
    ):
        (segment / filename).write_bytes(contents)
    config = dict(
        replans=1,
        case_wall_seconds=1,
        normal_target_N=1,
        cases=[
            dict(name=side, side=side, candidates=1, wrist_freedom_deg=0)
            for side in ("left", "right")
        ],
    )
    (working / "settings.json").write_text(json.dumps(config))
    runtime = tmp_path / "shared-runtime/source-id"
    runtime.mkdir(parents=True)
    archive = tmp_path / "shared-store/source-id.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"Source archive fixture")
    git = dict(revision="0" * 40, tree="1" * 40, dirty_sources=True, status=["?? source.py"])

    def write_reference(destination):
        Path(destination).write_text(json.dumps(dict(content_id="source-id", git=git)))

    frozen = SimpleNamespace(
        runtime_root=runtime,
        archive=archive,
        git=git,
        content_id="source-id",
        write_reference=write_reference,
        files={
            "src/abstract_stack/__init__.py": dict(
                sha256=hashlib.sha256(b"").hexdigest(), bytes=0, executable=False
            )
        },
    )
    calls = []

    def supervise(command, **kwargs):
        calls.append((command, kwargs))
        # A missing worker result remains unknown; it cannot become a success.
        return dict(
            returncode=1, timeout=False, wall_seconds=0, wall_budget_seconds=kwargs["wall_seconds"]
        )

    monkeypatch.setattr(source_store, "freeze_source", lambda *args, **kwargs: frozen)
    monkeypatch.setattr(checkpoint_manifest, "inspect_checkpoint", lambda path: (Path(path), {}))
    monkeypatch.setattr(module["runpy"], "run_path", lambda path: {"supervise": supervise})
    argv = [str(script), "--checkpoint", "fixture", "--output", "result"]
    if name == "render_checkpoint_chain":
        argv += ["--segments", "segment"]
    if name == "evaluate_squaring_focus":
        argv += ["--config", "settings.json"]
    monkeypatch.setattr(sys, "argv", argv)
    assert module["main"]() == 2
    output = working / "result"
    assert not list(output.rglob("*.py"))
    assert not (output / "source_snapshot").exists() and not (output / "frozen").exists()
    for command, invocation in calls:
        assert Path(command[1]).is_relative_to(runtime)
        assert invocation["environment"]["ABSTRACT_STACK_SOURCE_STORE"] == str(archive.parent)
        assert invocation["environment"]["ABSTRACT_STACK_SOURCE_REFERENCE"] == str(
            output / "source_ref.json"
        )
        assert invocation["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    manifest = json.loads((output / "manifest.json").read_text())
    hashes = manifest.get("sha256", manifest.get("input_sha256", {}))
    assert (
        hashes["source_ref.json"]
        == hashlib.sha256((output / "source_ref.json").read_bytes()).hexdigest()
    )
    if name == "evaluate_squaring_focus":
        assert len(calls) == 2
        for command, _ in calls:
            assert command[command.index("--horizon") + 1] == "0.2"
            assert command[command.index("--execute-prefix") + 1] == "0.1"
            assert command[command.index("--replans") + 1] == "1"
        report = json.loads((output / "report.json").read_text())
        assert report["scheduled_cases"] == 2
        assert all(case["outcome"] == "unknown" for case in report["results"])
    else:
        reference = output / (
            "checkpoint_ref.json" if name == "continue_stocking" else "inputs/checkpoint_ref.json"
        )
        recorded = json.loads(reference.read_text())
        assert Path(recorded["path"]).is_absolute() and Path(recorded["path"]) == checkpoint
        assert (
            recorded["sha256"]
            == hashlib.sha256((checkpoint / "manifest.json").read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("action", ["capture", "verify"])
def test_checkpoint_cli_parent_shares_sources_and_pins_absolute_inputs(
    tmp_path, monkeypatch, action
):
    from abstract_stack import checkpoint_cli

    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    fixture = working / "fixture"
    (fixture / "initial-scene").mkdir(parents=True)
    for name in (
        "manifest.json",
        "summary.json",
        "initial-scene/manifest.json",
        "initial-scene/scene.xml",
        "rollout_terminal.npz",
    ):
        (fixture / name).write_text("fixture input\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    archive = tmp_path / "bundles/source.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"source archive fixture")

    def write_reference(path):
        path.write_text('{"content_id":"fixture"}')

    frozen = SimpleNamespace(
        runtime_root=runtime,
        archive=archive,
        content_id="fixture",
        git=dict(revision="0" * 40, dirty_sources=True),
        files={"src/abstract_stack/__init__.py": dict(sha256="1" * 64)},
        write_reference=write_reference,
    )
    monkeypatch.setattr(source_store, "freeze_source", lambda *args, **kwargs: frozen)
    monkeypatch.setattr(checkpoint_manifest, "inspect_checkpoint", lambda path: (Path(path), {}))
    invocations = []

    class FailedWorker:
        returncode = 1

        def wait(self, **kwargs):
            return self.returncode

    def popen(command, **kwargs):
        invocations.append((command, kwargs))
        return FailedWorker()

    monkeypatch.setattr(checkpoint_cli.subprocess, "Popen", popen)
    parser = argparse.ArgumentParser()
    checkpoint_cli.add_commands(parser)
    argv = [action, "--output", "result"]
    argv += ["--episode", "fixture"] if action == "capture" else ["fixture"]
    args = parser.parse_args(argv)
    assert checkpoint_cli.run(args) == 2
    output = working / "result"
    assert not list(output.rglob("*.py")) and not (output / "source_snapshot").exists()
    reference = json.loads((output / "input_ref.json").read_text())
    assert Path(reference["path"]) == fixture and reference["kind"] == action
    for name, digest in reference["hashes"].items():
        assert digest == hashlib.sha256((fixture / name).read_bytes()).hexdigest()
    command, invocation = invocations[0]
    assert str(fixture) in command
    assert invocation["env"]["PYTHONPATH"] == str(runtime / "src")
    assert invocation["env"]["ABSTRACT_STACK_SOURCE_STORE"] == str(archive.parent)
    assert invocation["env"]["ABSTRACT_STACK_SOURCE_REFERENCE"] == str(output / "source_ref.json")
    assert invocation["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    manifest = json.loads((output / "capture_manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert digest == hashlib.sha256((output / name).read_bytes()).hexdigest()
    (fixture / "manifest.json").write_text("modified after capture\n")
    if action == "verify":
        args.worker = True
        assert checkpoint_cli.run(args) == 2
        assert "Referenced checkpoint input changed" in (output / "error.json").read_text()
    else:
        args.worker = True
        args.state = "start"
        assert checkpoint_cli.run(args) == 2
        assert "capture state differs" in (output / "error.json").read_text()
    (output / "input_ref.json").unlink()
    assert checkpoint_cli.run(args) == 2
    assert "input reference missing" in (output / "error.json").read_text()
