"""Source/config freezing and process failures cannot manufacture task credit."""

import hashlib
import json
import os
import runpy
import signal
import sys
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/run_frozen_replanning.py")
    )


def test_source_and_control_config_are_frozen_before_worker_launch(tmp_path, runner):
    root, snapshot = tmp_path / "checkout", tmp_path / "snapshot"
    (root / "src/abstract_stack").mkdir(parents=True)
    (root / "scripts").mkdir()
    source = root / "src/abstract_stack/policy.py"
    source.write_text("value = 1\n")
    (root / "scripts/run_replanning.py").write_text("print('worker fixture')\n")
    config = tmp_path / "controller.json"
    config.write_text('{"gain": 1}\n')
    arguments, hashes = runner["freeze_sources"](root, snapshot, ["--control-config", str(config)])
    source.write_text("value = 2\n")
    config.write_text('{"gain": 2}\n')
    assert (snapshot / "src/abstract_stack/policy.py").read_text() == "value = 1\n"
    frozen_config = Path(arguments[1])
    assert json.loads(frozen_config.read_text()) == {"gain": 1}
    assert hashes[frozen_config.name] == hashlib.sha256(frozen_config.read_bytes()).hexdigest()


def test_signal_failure_and_partial_success_summary_earn_zero_credit(tmp_path, runner):
    summary = dict(
        task_success=True,
        manipulation_evaluation=dict(
            scheduled_manipulations=2,
            stages=dict(pick=2, place=2, final_position=2, retreat=2),
        ),
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    result = runner["classify"](dict(returncode=-11, timeout=False), tmp_path, planning_only=False)
    assert result["verification_status"] == "unknown"
    assert not any(result["credited_stages"].values())
    assert json.loads(path.read_text()) == summary
    summary["manipulation_evaluation"]["stages"]["retreat"] = 1
    path.write_text(json.dumps(summary))
    result = runner["classify"](dict(returncode=0, timeout=False), tmp_path, planning_only=False)
    assert result["verification_status"] == "unknown"
    assert not result["task_success"]
    summary["task_success"] = False
    path.write_text(json.dumps(summary))
    result = runner["classify"](dict(returncode=2, timeout=False), tmp_path, planning_only=False)
    assert result["verification_status"] == "verified_terminal_outcome"
    assert result["credited_stages"]["retreat"] == 1


def test_successful_read_only_planning_is_never_a_manipulation_success(tmp_path, runner):
    (tmp_path / "plan_probe.json").write_text(
        json.dumps(dict(failure=None, live_state_unchanged=True))
    )
    result = runner["classify"](dict(returncode=0, timeout=False), tmp_path, planning_only=True)
    assert result["planning_succeeded"]
    assert not result["task_success"]
    assert not any(result["credited_stages"].values())


@pytest.mark.skipif(sys.platform != "linux", reason="Child zombie check uses Linux /proc")
def test_timeout_terminates_the_created_process_group_including_its_child(tmp_path, runner):
    pid_path = tmp_path / "child_pid"
    code = (
        "import subprocess,sys,time,pathlib; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid)); time.sleep(30)"
    )
    result = runner["supervise"](
        [sys.executable, "-c", code],
        cwd=tmp_path,
        environment=os.environ.copy(),
        log=tmp_path / "run.log",
        wall_seconds=0.5,
    )
    assert result["timeout"]
    assert result["timeout_reason"] == "wall"
    assert result["returncode"] == -signal.SIGKILL
    stat = Path(f"/proc/{int(pid_path.read_text())}/stat")
    try:
        state = stat.read_text().split(") ", 1)[1]
    except (FileNotFoundError, ProcessLookupError):
        # A killed child may be reaped between opening and reading /proc.
        state = None
    assert state is None or state.startswith("Z ")


def test_progress_timeout_uses_the_last_update_and_retains_evidence(tmp_path, runner):
    progress = tmp_path / "physics-progress.json"
    code = f"""\
import pathlib
import time
progress = pathlib.Path({str(progress)!r})
for step in range(6):
    progress.write_text(str(step))
    time.sleep(0.15)
time.sleep(30)
"""
    result = runner["supervise"](
        [sys.executable, "-c", code],
        cwd=tmp_path,
        environment=os.environ.copy(),
        log=tmp_path / "run.log",
        wall_seconds=5,
        progress_path=progress,
        idle_seconds=0.7,
    )
    assert result["timeout"]
    assert result["timeout_reason"] == "progress_idle"
    assert result["returncode"] == -signal.SIGKILL
    assert progress.read_text() == "5"
