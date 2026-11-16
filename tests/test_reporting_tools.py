"""A native process failure must not erase scheduled dense manipulations."""

import json
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("terminal", ["{partial", "[]"])
def test_dense_collector_preserves_denominator_with_broken_terminal_json(tmp_path, terminal):
    collect = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/report_dense_runs.py")
    )["collect"]
    (tmp_path / "initial-scene").mkdir()
    (tmp_path / "initial-scene/manifest.json").write_text(
        json.dumps({"config": {"product_count": 10, "initially_stocked": 9}})
    )
    (tmp_path / "summary.json").write_text(terminal)
    (tmp_path / "process_result.json").write_text(json.dumps({"returncode": -11}))
    result = collect(tmp_path)
    assert result["scheduled_manipulations"] == 1
    assert not result["verified_terminal_outcome"]
    assert not result["clean_success"]
    assert result["artifact_error"]
    assert not any(result["stages"].values())
