"""Compact plan, execution and recovery history beside the native artifacts."""

import json
from pathlib import Path

HISTORY_NAME = "agent-history.jsonl"
CURSOR_FIELDS = (
    "sequence",
    "product",
    "skill",
    "phase",
    "carton_step",
    "skill_step",
    "case_id",
    "action",
    "branch",
    "attempt",
    "sim_time_s",
    "artifacts",
)


class AgentHistory:
    """One writer per run. Each complete line survives an interrupted rollout."""

    def __init__(self, directory, *, run_kind, **metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / HISTORY_NAME
        self.sequence = 0
        header = dict(
            schema_version=1, sequence=0, event="run_started", run_kind=run_kind, **metadata
        )
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(header, allow_nan=False) + "\n")

    def record(self, event, **details):
        sequence = self.sequence + 1
        entry = dict(sequence=sequence, event=event, **details)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, allow_nan=False, separators=(",", ":")) + "\n")
        self.sequence = sequence
        return sequence


def history_path(path):
    path = Path(path)
    if path.is_dir():
        candidate = path / HISTORY_NAME
        path = candidate if candidate.is_file() else path / "episode" / HISTORY_NAME
    return path


def read_history(path):
    """Read a live journal through its last complete record."""
    with history_path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.endswith("\n"):
                break
            entry = json.loads(line)
            if entry.get("sequence") != index or (
                index == 0
                and (entry.get("schema_version"), entry.get("event")) != (1, "run_started")
            ):
                raise ValueError(f"Invalid agent history record at sequence {index}")
            yield entry


def _product_evidence():
    return dict(skills_used=[], skills_verified=[], attempts=[], gates={}, plan_steps=[])


def summarize_history(entries):
    """Reduce the journal to a current cursor and compact per-carton evidence."""
    entries = iter(entries)
    header = next(entries, None)
    if header is None:
        raise ValueError("Agent history is empty")
    products = {name: _product_evidence() for name in header.get("scheduled_products", ())}
    cursor, checkpoints, decisions, finished = {}, {}, [], None
    sequence = 0
    for row in entries:
        event, product = row["event"], row.get("product")
        sequence = row["sequence"]
        if product is not None:
            evidence = products.setdefault(product, _product_evidence())
            skill = row.get("skill")
            phase_is_execution = (
                event == "phase_started"
                and not cursor.get("selecting")
                and row.get("phase") != "insertion_mode"
            )
            if skill and (phase_is_execution or event in {"skill_started", "attempt_started"}):
                if skill not in evidence["skills_used"]:
                    evidence["skills_used"].append(skill)
            if event == "carton_finished":
                evidence["gates"].update(row["gates"])
                evidence["carton_outcome"] = row["outcome"]
            if event == "carton_plan":
                evidence["plan_steps"] = row["steps"]
            if event in {
                "attempt_finished",
                "skill_finished",
                "execution_failed",
                "reset_rejected",
            }:
                evidence["attempts"].append(row)
                if (
                    skill
                    and row.get("outcome") == "verified"
                    and skill not in evidence["skills_verified"]
                ):
                    evidence["skills_verified"].append(skill)
            if event == "checkpoint_saved":
                checkpoints[product] = row
        if event in {
            "carton_selection_started",
            "carton_selected",
            "reset_requested",
            "attempt_started",
            "skill_started",
            "phase_started",
            "branch_started",
        }:
            if event in {"carton_selection_started", "reset_requested", "branch_started"}:
                cursor.clear()
                cursor["selecting"] = event == "carton_selection_started"
            if event in {"carton_selected", "attempt_started", "skill_started"}:
                cursor["selecting"] = False
            if event in {"attempt_started", "skill_started", "carton_selected"}:
                cursor.pop("phase", None)
            if row.get("skill") != cursor.get("skill"):
                cursor.pop("skill_step", None)
            cursor.update({key: row[key] for key in CURSOR_FIELDS if key in row})
            if (
                event == "phase_started"
                and product is not None
                and cursor.get("skill_step") is None
            ):
                plan_steps = products[product]["plan_steps"]
                if row.get("skill") in plan_steps:
                    cursor["skill_step"] = 1 + plan_steps.index(row["skill"])
        if event in {
            "carton_selection_started",
            "carton_selected",
            "carton_plan",
            "plan_started",
            "attempt_started",
            "branch_started",
        }:
            decisions.append(row)
        if event == "run_finished":
            finished = row
            for name, gates in row.get("gates", {}).items():
                products.setdefault(name, _product_evidence())["gates"].update(gates)
    for product in products.values():
        product["unreached_steps"] = [
            skill for skill in product["plan_steps"] if skill not in product["skills_used"]
        ]
    return dict(
        run_kind=header["run_kind"],
        status=finished["outcome"] if finished else "unfinished",
        last_sequence=sequence,
        current=cursor,
        products=products,
        decisions=decisions,
        latest_checkpoints=checkpoints,
        terminal=finished,
        scheduled_products=header.get("scheduled_products", []),
        prefilled_products=header.get("prefilled_products", []),
    )


def inspect_history(path):
    """Include the supervisor's exit evidence when the run used one."""
    journal = history_path(path)
    status = summarize_history(read_history(journal))
    parent = journal.parent
    process_path = (parent.parent if parent.name == "episode" else parent) / "process.json"
    if process_path.is_file():
        try:
            process = json.loads(process_path.read_text())
        except json.JSONDecodeError:
            return status  # The supervisor can be writing its result right now.
        status["process"] = {
            key: process.get(key) for key in ("outcome", "returncode", "timeout", "wall_seconds")
        }
        code = process.get("returncode")
        if (
            process.get("timeout")
            or code is not None
            and (code not in (0, 2) or status["terminal"] is None)
        ):
            status["status"] = "process_failure_or_unknown"
    return status
