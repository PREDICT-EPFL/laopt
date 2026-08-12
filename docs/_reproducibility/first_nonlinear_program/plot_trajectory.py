#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def read_trajectory(path: Path) -> dict[str, list[float]]:
    columns = {
        "time": [],
        "angle": [],
        "angular_velocity": [],
        "torque": [],
    }
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            for name in columns:
                columns[name].append(float(row[name]))
    return columns


def plot_trajectory(data: dict[str, list[float]], output: Path) -> None:
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "savefig.bbox": "tight",
    })

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 5.8),
        sharex=True,
        constrained_layout=True,
    )

    time = data["time"]
    axes[0].plot(time, data["angle"], color="#0072B2", linewidth=2)
    axes[0].axhline(0.0, color="#555555", linewidth=1, linestyle="--")
    axes[0].set_ylabel(r"Angle $\theta$ [rad]")

    axes[1].plot(
        time,
        data["angular_velocity"],
        color="#D55E00",
        linewidth=2,
    )
    axes[1].axhline(0.0, color="#555555", linewidth=1, linestyle="--")
    axes[1].set_ylabel(r"Velocity $\omega$ [rad/s]")

    axes[2].step(
        time,
        data["torque"],
        where="post",
        color="#009E73",
        linewidth=2,
    )
    axes[2].axhline(3.0, color="#777777", linewidth=1, linestyle=":")
    axes[2].axhline(-3.0, color="#777777", linewidth=1, linestyle=":")
    axes[2].set_ylabel(r"Torque $u$ [N m]")
    axes[2].set_xlabel("Time [s]")

    for axis in axes:
        axis.grid(axis="y", color="#d8d8d8", linewidth=0.7)
        axis.set_xlim(time[0], time[-1])

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir / "trajectory.csv"
    output_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else script_dir.parent.parent / "assets/images/inverted_pendulum_solution.svg"
    )
    plot_trajectory(read_trajectory(csv_path), output_path)
