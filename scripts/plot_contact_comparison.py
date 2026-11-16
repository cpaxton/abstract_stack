#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["matplotlib==3.11.2", "numpy>=1.24,<2"]
# ///
"""Plot the committed 20 ms contact evidence without simulator dependencies.

Run with ``uv run scripts/plot_contact_comparison.py`` for isolated plotting
dependencies, or use an existing environment containing the pinned Matplotlib.
Each force sample is the maximum of its original 2 ms measurements in that bin.
"""

import argparse
import csv
from pathlib import Path

import matplotlib


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=Path("paper/evidence/contact_force_comparison.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/images/contact_force_comparison.png")
    )
    args = parser.parse_args()
    with args.csv.open() as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5.8), layout="constrained")
    for key, label, color in [
        ("legacy", "Legacy: force-limit abort", "#bf4539"),
        ("selected", "Selected: completed shelf", "#286da8"),
    ]:
        data = [r for r in rows if r["policy"] == key]
        times = [float(r["time_s"]) for r in data]
        axes[0].plot(
            times,
            [float(r["arm_fixture_peak_N"]) for r in data],
            label=label,
            color=color,
            linewidth=1.7,
        )
        axes[1].plot(times, [float(r["back_gap_mm"]) for r in data], color=color, linewidth=1.7)
    release = min(
        float(r["time_s"]) for r in rows if r["policy"] == "selected" and r["phase"] != "push"
    )
    end = max(float(r["time_s"]) for r in rows)
    for ax in axes:
        ax.axvspan(release, end, color="0.85", alpha=0.45)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(25, color="#bf4539", linestyle="--", linewidth=1, label="25 N abort threshold")
    axes[0].set_ylabel("Arm–fixture normal force (N)")
    axes[0].set_ylim(-0.5, 30)
    axes[0].legend(loc="center right", frameon=False)
    axes[0].set_title("Same seeded scene: wrist clearance prevents the insertion contact abort")
    axes[1].axhline(0, color="0.3", linestyle=":", linewidth=1)
    axes[1].set_ylabel("Carton–shelf-back gap (mm)")
    axes[1].set_xlabel("Simulation seconds since second-carton push began")
    axes[1].text(
        release + 0.2,
        0.93,
        "Release and retreat",
        transform=axes[1].get_xaxis_transform(),
        va="top",
        color="0.35",
        fontsize=9,
    )
    fig.suptitle("Native MABEL / MuJoCo • scene 8 • 20 ms bins preserve force peaks", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
