"""Serial scene/solver/video campaigns with explicit failed and crashed runs."""

from __future__ import annotations

import json
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
    if not summary_path.exists():
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


def run_campaign(directory, seeds, *, objects=2, video=False, fps=15):
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("campaign seeds must be distinct nonnegative integers")
    if not 1 <= objects <= 6:
        raise ValueError("campaigns support 1 through 6 cartons")
    if not isinstance(fps, int) or not 1 <= fps <= 60:
        raise ValueError("campaign video fps must be an integer between 1 and 60")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds),
        "objects": objects,
        "video": video,
        "fps": fps,
        "policy": "world_state_stocking",
        "execution": "serial isolated processes",
        "runs": [],
    }
    # Preserve earlier campaigns and all their episode directories.
    index = 1
    manifest_path = root / "campaign.json"
    while manifest_path.exists():
        manifest_path = root / f"campaign-{index}.json"
        index += 1
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
        manifest["active_run"] = {"seed": seed, "directory": str(episode)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        with (episode / "run.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result = summarize_run(episode, process.returncode, video=video)
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
