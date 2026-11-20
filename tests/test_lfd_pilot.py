"""Bounded collection scheduling preserves failures and excludes held-out cases."""

import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def case(tmp_path, index, *, split="development"):
    directory = tmp_path / f"case-{index}"
    directory.mkdir()
    config = directory / "scene.json"
    config.write_text(json.dumps(dict(product_count=2 + index, levels=[0.7, 1.0])))
    (directory / "context.json").write_text(
        json.dumps(
            dict(
                suite_version="stocking_geometry_dr_v1",
                split=split,
                scene_seed=1200 + index,
                reset_seed=index,
                scene_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            )
        )
    )
    return directory


def test_heldout_case_rejected_before_freezing_or_creating_outputs(tmp_path, monkeypatch):
    main = runpy.run_path(str(Path(__file__).parents[1] / "scripts/collect_lfd_pilot.py"))["main"]
    heldout = case(tmp_path, 0, split="heldout")
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["collect", "--case", str(heldout), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and not output.exists()


def test_both_scheduled_failures_remain_reported_with_shared_sources(tmp_path, monkeypatch):
    import abstract_stack.source_store as sources

    main = runpy.run_path(str(Path(__file__).parents[1] / "scripts/collect_lfd_pilot.py"))["main"]
    cases = [case(tmp_path, i) for i in range(2)]
    output = tmp_path / "output"
    freezes, commands = [], []
    archive = tmp_path / "shared" / "bundle.zip"
    runtime = tmp_path / "runtime"

    def freeze(root, *, paths):
        freezes.append(paths)
        return SimpleNamespace(
            content_id="fixed",
            runtime_root=runtime,
            archive=archive,
            git=dict(revision="git-head", dirty_sources=True),
            write_reference=lambda path: path.write_text('{"content_id":"fixed"}'),
        )

    def supervise(command, **kwargs):
        commands.append(command)
        assert kwargs["environment"]["ABSTRACT_STACK_SOURCE_STORE"] == str(archive.parent)
        assert kwargs["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
        kwargs["log"].write_text("retained simulated process failure")
        return dict(
            returncode=-11 if len(commands) == 1 else 2,
            timeout=False,
            wall_seconds=0.1,
            wall_budget_seconds=kwargs["wall_seconds"],
        )

    monkeypatch.setattr(sources, "freeze_source", freeze)
    monkeypatch.setattr(runpy, "run_path", lambda path: {"supervise": supervise})
    monkeypatch.setattr(
        sys,
        "argv",
        ["collect", "--case", str(cases[0]), "--case", str(cases[1]), "--output", str(output)],
    )
    assert main() == 2
    report = json.loads((output / "report.json").read_text())
    assert len(freezes) == 1 and len(commands) == report["scheduled_cases"] == 2
    assert [r["outcome"] for r in report["results"]] == ["unknown", "unknown"]
    assert not report["task_success"] and not report["training_eligible"]
    assert not list(output.rglob("*.py"))
