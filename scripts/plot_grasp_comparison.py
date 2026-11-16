#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["matplotlib==3.11.2", "numpy>=1.24,<2"]
# ///
"""Plot measured approach evidence from the committed peak-preserving CSV."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib


def extract_evidence(baseline, selected, output):
    """Retain every physics-step force peak while binning at the control period."""
    import numpy as np

    rows = []
    for policy, directory in (("neutral", baseline), ("preshape", selected)):
        summary = json.loads((directory / "summary.json").read_text())
        events = summary["events"]
        approach = next(e for e in events if e["phase"] == "approach")
        start = approach["sim_time"]
        end = next((e["sim_time"] for e in events if e["phase"] == "lift"), float("inf"))
        columns = summary["trace_columns"]
        with np.load(directory / "trace.npz") as trace:
            samples = trace["samples"]
            samples = samples[(samples[:, 0] > start) & (samples[:, 0] <= end)]
        bins = np.floor((samples[:, 0] - start) / 0.02 + 1e-7).astype(int)
        for index in np.unique(bins):
            group = samples[bins == index]
            force = group[:, columns.index("hand_normal_N")]
            force = force + group[:, columns.index("arm_product_normal_N")]
            last = group[-1]
            phase = next(e["phase"] for e in reversed(events) if e["sim_time"] < last[0])
            rows.append(
                {
                    "policy": policy,
                    "time_s": float(last[0] - start),
                    "phase": phase,
                    "robot_product_peak_N": float(force.max()),
                    "hand_above_box_center_mm": float((last[6] - last[3]) * 1000),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=Path("paper/evidence/thumb_approach_comparison.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/images/thumb_approach_comparison.png")
    )
    parser.add_argument("--baseline", type=Path, help="Optional native no-preshape run directory")
    parser.add_argument("--selected", type=Path, help="Optional native preshape run directory")
    args = parser.parse_args()
    if bool(args.baseline) != bool(args.selected):
        parser.error("--baseline and --selected must be supplied together")
    if args.baseline:
        extract_evidence(args.baseline, args.selected, args.csv)
    with args.csv.open() as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5.5), layout="constrained")
    for policy, label, color in (
        ("neutral", "Neutral thumb: approach abort", "#bf4539"),
        ("preshape", "Thumb MCP 0.4 rad: full scene completed", "#286da8"),
    ):
        data = [r for r in rows if r["policy"] == policy]
        time = [float(r["time_s"]) for r in data]
        axes[0].plot(
            time,
            [float(r["robot_product_peak_N"]) for r in data],
            color=color,
            label=label,
            linewidth=1.7,
        )
        axes[1].plot(
            time,
            [float(r["hand_above_box_center_mm"]) for r in data],
            color=color,
            linewidth=1.7,
        )
    axes[0].axhline(
        25, color="#bf4539", linestyle="--", linewidth=1, label="Unchanged 25 N abort threshold"
    )
    axes[0].set_ylabel("Robot–carton normal force (N)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Opening the thumb base clears the carton's top edge")
    axes[1].set_ylabel("Hand point above carton center (mm)")
    axes[1].set_xlabel("Simulation seconds since first approach began")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Native MABEL / MuJoCo • development scene 112 • 20 ms force maxima", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
