"""Concurrent curriculum cases keep their schedule and independent outcomes."""

import json
import runpy
import sys
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from abstract_stack import source_store


def test_parallel_ledger_preserves_schedule_and_retains_process_failures(tmp_path, monkeypatch):
    pytest.importorskip("gymnasium")
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_skill_curriculum.py"
    launcher = runpy.run_path(str(script))
    output = tmp_path / "run"
    frozen = SimpleNamespace(
        runtime_root=tmp_path / "runtime",
        archive=tmp_path / "store" / "bundle.zip",
        content_id="frozen",
        write_reference=lambda path: path.write_text('{"content_id": "frozen"}'),
    )
    overlap = Barrier(2)

    def supervise(command, *, log, environment, wall_seconds, stop_event, **kwargs):
        assert stop_event is not None
        assert wall_seconds == 1
        assert environment["ABSTRACT_STACK_SOURCE_REFERENCE"] == str(output / "source_ref.json")
        assert environment["MABEL_ROOT"] == str(script.parents[1] / "third_party/open_mabel")
        seed = int(command[command.index("--seed") + 1])
        log.write_text(f"worker {seed}")
        (log.parent / "skill-result.json").write_text('{"outcome": "verified"}')
        if seed in (3, 7):
            # A serial dispatcher would break this barrier.
            overlap.wait(timeout=5)
        if seed == 7:
            return dict(returncode=-9, timeout=True, timeout_reason="wall")
        if seed == 5:
            raise OSError("worker launch failed")
        return dict(returncode=0, timeout=False)

    monkeypatch.setattr(source_store, "freeze_source", lambda *args, **kwargs: frozen)
    monkeypatch.setattr(launcher["runpy"], "run_path", lambda path: {"supervise": supervise})
    monkeypatch.delenv("MABEL_ROOT", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--gym",
            "--skill",
            "push",
            "--seed",
            "3",
            "--seed",
            "7",
            "--seed",
            "5",
            "--workers",
            "2",
            "--case-wall-seconds",
            "1",
            "--output",
            str(output),
        ],
    )
    assert launcher["main"]() == 0
    report = json.loads((output / "report.json").read_text())
    assert [case["seed"] for case in report["cases"]] == [3, 7, 5]
    assert [case["outcome"] for case in report["cases"]] == [
        "verified",
        "process_failure_or_unknown",
        "process_failure_or_unknown",
    ]
    assert report["verified"] == 1 and report["fresh_task_credit"] == 0
    assert report["workers"] == 2
    assert report["case_dispatch_wall_s"] > 0
    assert not (output / "report.json.tmp").exists()
    for case in report["cases"]:
        directory = (output / case["result"]).parent
        assert (directory / "run.log").read_text() == f"worker {case['seed']}"
        assert json.loads((directory / "process.json").read_text()) == case["process"]
