"""Serial scene/solver/video campaigns with explicit failed and crashed runs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def summarize_run(directory, returncode, *, video):
    directory = Path(directory)
    summary_path = directory / "summary.json"
    result = {
        "directory": str(directory),
        "returncode": returncode,
        "verification_status": "unknown",
        "physical_success": None,
        "artifacts_valid": False,
    }
    process_failure = directory / "process_failure.json"
    if process_failure.exists():
        result["process_failure"] = json.loads(process_failure.read_text())
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = str(-returncode)
        result["process_failure"] = {
            **result.get("process_failure", {}),
            "kind": "native_signal",
            "signal": name,
            "returncode": returncode,
        }
    if not summary_path.exists():
        if "process_failure" not in result:
            result["process_failure"] = {
                "kind": "missing_terminal_summary",
                "returncode": returncode,
                "verification_status": "unknown",
            }
        result["failure"] = "Process ended without a terminal summary; inspect run.log"
        return result
    try:
        summary = json.loads(summary_path.read_text())
        result.update(
            verification_status="verified",
            physical_success=summary["physical_success"],
            failure=summary.get("failure"),
            stocked_count=summary["metrics"]["stocked_count"],
            product_count=summary["metrics"]["product_count"],
            elapsed_sim_time=summary["metrics"]["elapsed_sim_time"],
            stage_evaluation=summary.get("stage_evaluation"),
            peak_forces_N=summary.get("peak_forces_N"),
        )
        result["artifacts_valid"] = all(
            (directory / name).exists() for name in ("trace.npz", "rollout_start.npz")
        )
        if video:
            metadata = json.loads((directory / "videos/video.json").read_text())
            result["artifacts_valid"] = (
                result["artifacts_valid"]
                and all(
                    (directory / f"videos/{name}.mp4").is_file()
                    for name in ("first_person", "third_person")
                )
                and metadata["physical_success"] == summary["physical_success"]
            )
    except (OSError, ValueError, KeyError) as error:
        result["artifacts_valid"] = False
        result["artifact_error"] = str(error)
    return result


def run_campaign(
    directory,
    seeds,
    *,
    objects=2,
    video=False,
    fps=15,
    control_config=None,
    episode_timeout=None,
    legacy_contact_control=False,
    guarded_approach=False,
    fixture_avoidance=True,
    thumb_preshape=True,
):
    from abstract_stack.provenance import policy_provenance
    from abstract_stack.tuning import load_control_config

    controls = load_control_config(control_config)
    if not isinstance(legacy_contact_control, bool):
        raise ValueError("legacy_contact_control must be boolean")
    if not isinstance(thumb_preshape, bool):
        raise ValueError("thumb_preshape must be boolean")
    if not isinstance(guarded_approach, bool) or not isinstance(fixture_avoidance, bool):
        raise ValueError("guarded_approach and fixture_avoidance must be boolean")
    if episode_timeout is None:
        episode_timeout = 900 if video else 180
    if not isinstance(episode_timeout, (float, int)) or not 0 < episode_timeout < float("inf"):
        raise ValueError("episode_timeout must be finite and positive")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("campaign seeds must be distinct nonnegative integers")
    if not 1 <= objects <= 6:
        raise ValueError("campaigns support 1 through 6 cartons")
    if not isinstance(fps, int) or not 1 <= fps <= 60:
        raise ValueError("campaign video fps must be an integer between 1 and 60")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy_source": policy_provenance(),
        "format_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds),
        "objects": objects,
        "video": video,
        "fps": fps,
        "policy": "world_state_stocking",
        "execution": "serial isolated processes",
        "runs": [],
        "control_overrides": controls,
        "episode_wall_timeout_seconds": episode_timeout,
        "contact_aware": not legacy_contact_control,
        "guarded_approach": guarded_approach,
        "fixture_avoidance": fixture_avoidance,
        "thumb_preshape": thumb_preshape,
    }
    # Preserve earlier campaigns and all their episode directories.
    index = 1
    manifest_path = root / "campaign.json"
    while manifest_path.exists():
        manifest_path = root / f"campaign-{index}.json"
        index += 1
    controls_path = manifest_path.with_suffix(".controls.json")
    controls_path.write_text(json.dumps(controls, indent=2) + "\n")
    for seed in seeds:
        episode = root / f"seed-{seed}"
        attempt = 1
        while episode.exists():
            episode = root / f"seed-{seed}-attempt-{attempt}"
            attempt += 1
        episode.mkdir()
        command = [
            sys.executable,
            "-m",
            "abstract_stack.cli",
            "solve",
            "--random-scene",
            "--objects",
            str(objects),
            "--seed",
            str(seed),
            "--output-dir",
            str(episode),
        ]
        if video:
            command += ["--video", "--fps", str(fps)]
        if legacy_contact_control:
            command += ["--legacy-contact-control"]
        if guarded_approach:
            command += ["--guarded-approach"]
        command += ["--fixture-avoidance" if fixture_avoidance else "--no-fixture-avoidance"]
        command += ["--thumb-preshape" if thumb_preshape else "--no-thumb-preshape"]
        command += ["--control-config", str(controls_path.resolve())]
        manifest["active_run"] = {"seed": seed, "directory": str(episode)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        with (episode / "run.log").open("w") as log:
            try:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=episode_timeout,
                    env=dict(os.environ, PYTHONFAULTHANDLER="1"),
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                failure = {
                    "kind": "wall_timeout",
                    "wall_timeout_seconds": episode_timeout,
                    "verification_status": "unknown",
                }
                (episode / "process_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        result = summarize_run(episode, returncode, video=video)
        if result.get("process_failure") and not (episode / "process_failure.json").exists():
            (episode / "process_failure.json").write_text(
                json.dumps(result["process_failure"], indent=2) + "\n"
            )
        result["seed"] = seed
        manifest["runs"].append(result)
        print(json.dumps(result), flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest.pop("active_run", None)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["successful_runs"] = sum(
        r["physical_success"] is True and r["returncode"] == 0 and r["artifacts_valid"]
        for r in manifest["runs"]
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
