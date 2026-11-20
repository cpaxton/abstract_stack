"""Bounded controller search; task acceptance and force abort limits stay fixed."""

from __future__ import annotations

import json
from pathlib import Path

from abstract_stack.evaluation import reported_manipulations

# These ranges change compliance, timing and motion-reference staging only.
# They never change the task, IK acceptance, contact verification, displacement
# guards or force abort limit.
CONTROL_BOUNDS = {
    "grasp_force_target": (3.0, 10.0),
    "push_force_target": (0.5, 5.0),
    "grip_force_gain": (0.01, 0.10),
    "grip_max_rate": (0.10, 0.50),
    "push_speed": (0.01, 0.04),
    "insertion_advance_speed_m_s": (0.02, 0.065),
    "post_square_pusher_standoff_m": (0.02, 0.045),
    "base_speed_m_s": (0.015, 0.05),
    "payload_orientation_speed": (0.10, 0.60),
    "insertion_vertical_speed": (0.003, 0.02),
    "release_seconds": (2.0, 6.0),
    "approach_force_target": (5.0, 20.0),
    "contact_grasp_seconds": (4.0, 10.0),
    "contact_descent_speed": (0.005, 0.02),
    "fixture_force_target": (1.0, 8.0),
    "fixture_unload_speed": (0.005, 0.03),
    "fixture_unload_distance": (0.005, 0.025),
    "fixture_clearance": (0.002, 0.012),
    "release_lateral_limit": (0.01, 0.06),
    "alignment_timeout_seconds": (6.0, 14.0),
    "front_staging_depth": (0.01, 0.12),
    "insertion_yaw_lever_gain": (0.05, 0.6),
    "insertion_back_force_target": (0.3, 5.0),
    "insertion_lateral_position_gain": (0.2, 3.0),
    "insertion_tangent_admittance": (0.004, 0.04),
    "front_alignment_position_gain_s": (0.5, 8.0),
    "front_alignment_yaw_gain_s": (0.5, 8.0),
}


def load_control_config(path):
    if path is None:
        return {}
    settings = json.loads(Path(path).read_text())
    if not isinstance(settings, dict):
        raise ValueError("Control configuration must be a JSON object")
    for name, value in settings.items():
        if name not in CONTROL_BOUNDS:
            raise ValueError(f"Not a tunable compliant-control setting: {name}")
        lo, hi = CONTROL_BOUNDS[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be a number within [{lo}, {hi}]")
    return settings


def score_campaign(campaign):
    """Lexicographic: clean scenes, fewer unknowns/incidents, stages, force/time.

    Unscheduled or unknown stages receive zero. A process crash never earns
    progress, and no improvement in speed/force can compensate for lost success.
    """
    clean, unknown, knocked, stages, peak, seconds = 0, 0, 0, 0, 0.0, 0.0
    for run in campaign["runs"]:
        evaluation = run.get("stage_evaluation") or {}
        valid = run["returncode"] in (0, 2) and run.get("artifacts_valid", False)
        unknown += int(not valid or not evaluation)
        knocked += int(bool(evaluation.get("incidents")))
        grade = reported_manipulations(run)
        clean += int(valid and grade["clean_success"])
        if valid:
            stages += sum(grade["stages"].values())
            peak += max((run.get("peak_forces_N") or {}).values(), default=0)
        seconds += run.get("elapsed_sim_time", 240)
    return clean, -unknown, -knocked, stages, -round(peak, 4), -round(seconds, 4)


def hill_climb(directory, seeds, *, rounds=1, objects=2):
    """Coordinate hill-climb on a fixed development suite; keep every candidate.

    Each neighbour changes one setting from the incumbent by a factor. The
    starting policy and every candidate run on exactly the same seeds. Search
    never uses validation seeds; evaluation is a separate explicit campaign.
    """
    from abstract_stack.campaign import run_campaign
    from abstract_stack.interactions import ScriptConfig

    root = Path(directory)
    if root.exists():
        raise ValueError("Use a fresh hill-climb output directory")
    if not isinstance(rounds, int) or not 1 <= rounds <= 3:
        raise ValueError("hill-climb rounds must be 1 through 3")
    root.mkdir(parents=True)
    record = {
        "score_version": 2,
        "method": "coordinate hill climb",
        "development_seeds": list(seeds),
        "objects": objects,
        "trials": [],
        "acceptance_thresholds": "fixed",
    }
    incumbent = {}
    best_score = None
    candidates = [(None, 1.0)]
    # Both directions matter: increasing squeeze can prevent slip but increase
    # release drag, and faster uprighting can improve tilt or destabilize contact.
    neighbours = [
        (key, factor)
        for key in ("grasp_force_target", "push_force_target", "payload_orientation_speed")
        for factor in (0.75, 1.25)
    ]
    trial = 0
    for round_index in range(rounds):
        if round_index == 0:
            candidates += neighbours
        else:
            candidates = neighbours
        accepted = False
        for key, factor in candidates:
            candidate = dict(incumbent)
            if key is not None:
                lo, hi = CONTROL_BOUNDS[key]
                candidate[key] = min(
                    hi, max(lo, candidate.get(key, getattr(ScriptConfig(), key)) * factor)
                )
            path = root / f"trial-{trial:02d}.json"
            path.write_text(json.dumps(candidate, indent=2) + "\n")
            campaign = run_campaign(
                root / f"trial-{trial:02d}", seeds, objects=objects, control_config=path
            )
            score = score_campaign(campaign)
            improved = best_score is None or score > best_score
            record["trials"].append(
                {
                    "trial": trial,
                    "round": round_index,
                    "controls": candidate,
                    "score": score,
                    "accepted": improved,
                    "campaign": str(root / f"trial-{trial:02d}/campaign.json"),
                }
            )
            if improved:
                best_score, incumbent, accepted = score, candidate, True
            record.update(best_controls=incumbent, best_score=best_score)
            (root / "search.json").write_text(json.dumps(record, indent=2) + "\n")
            (root / "best-controls.json").write_text(json.dumps(incumbent, indent=2) + "\n")
            trial += 1
        if not accepted:
            break
    return record


def audit_search(directory):
    """Re-score retained trials; never rewrite the original experiment record."""
    root = Path(directory)
    record = json.loads((root / "search.json").read_text())
    best, controls, rows = None, None, []
    for trial in record["trials"]:
        campaign = json.loads(Path(trial["campaign"]).read_text())
        score = score_campaign(campaign)
        accepted = best is None or score > best
        rows.append(
            {
                "trial": trial["trial"],
                "score": score,
                "accepted": accepted,
                "original_accepted": trial["accepted"],
            }
        )
        if accepted:
            best, controls = score, trial["controls"]
    result = {
        "score_version": 2,
        "trials": rows,
        "best_controls": controls,
        "acceptance_decisions_unchanged": all(
            row["accepted"] == row["original_accepted"] for row in rows
        ),
        "winner_unchanged": controls == record["best_controls"],
    }
    (root / "search-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
