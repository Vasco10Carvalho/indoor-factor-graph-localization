#!/usr/bin/env python3
"""Align recorded trajectories at their initial pose and plot them on a ROS map."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from comparison_common import (
    Trajectory,
    align_initial_pose,
    load_map_metadata,
    load_trajectory,
    trajectory_metrics,
)


COLORS = {
    "Factor Graph 2D": "#004896",
    "AMCL": "#9c27b0",
    "SLAM Toolbox Localization": "#00acc1",
    "Cartographer Pure Localization": "#43a047",
    "KISS-ICP + Cartographer Pure Localization": "#e53935",
    "KISS-ICP Odometry": "#ff9800",
    "SLAM Toolbox": "#00acc1",
    "KISS-ICP": "#ff9800",
    "Cartographer": "#43a047",
    "KISS-ICP + Cartographer": "#e53935",
}


def parse_trajectory_specification(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use the format 'Label=/path/to/trajectory.csv'")
    label, raw_path = text.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Both label and CSV path are required")
    return label.strip(), Path(raw_path).expanduser()


def color_for(label: str, index: int) -> str:
    fallback = plt.get_cmap("tab10")
    return COLORS.get(label, fallback(index % 10))


def write_aligned_csv(path: Path, trajectories: list[Trajectory]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("source", "stamp_ns", "x", "y", "yaw_rad", "yaw_deg"),
        )
        writer.writeheader()
        for trajectory in trajectories:
            for stamp_ns, x, y, yaw in zip(
                trajectory.stamp_ns, trajectory.x, trajectory.y, trajectory.yaw_rad
            ):
                writer.writerow(
                    {
                        "source": trajectory.label,
                        "stamp_ns": int(stamp_ns),
                        "x": f"{x:.9f}",
                        "y": f"{y:.9f}",
                        "yaw_rad": f"{yaw:.12f}",
                        "yaw_deg": f"{math.degrees(yaw):.9f}",
                    }
                )


def make_plot(
    output: Path,
    map_yaml: Path,
    trajectories: list[Trajectory],
    dpi: int,
    title: str,
) -> None:
    metadata = load_map_metadata(map_yaml)
    image = cv2.imread(str(metadata.image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read map image: {metadata.image_path}")
    if abs(metadata.origin_yaw) > 1e-9:
        raise NotImplementedError("Plotting maps with a rotated YAML origin is not supported")

    height, width = image.shape
    extent = (
        metadata.origin_x,
        metadata.origin_x + width * metadata.resolution,
        metadata.origin_y,
        metadata.origin_y + height * metadata.resolution,
    )
    map_aspect = width / height
    if map_aspect < 0.8:
        figure_size = (max(7.0, 12.0 * map_aspect + 2.0), 12.0)
    elif map_aspect > 1.4:
        figure_size = (18.0, 9.5)
    else:
        figure_size = (12.0, 10.0)
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
    axis.imshow(image, cmap="gray", origin="upper", extent=extent, vmin=0, vmax=255)

    for index, trajectory in enumerate(trajectories):
        color = color_for(trajectory.label, index)
        linewidth = 2.7 if trajectory.label == "Factor Graph 2D" else 1.8
        zorder = 10 if trajectory.label == "Factor Graph 2D" else 6
        axis.plot(
            trajectory.x,
            trajectory.y,
            color=color,
            linewidth=linewidth,
            alpha=0.95,
            label=trajectory.label,
            zorder=zorder,
        )
        axis.scatter(
            trajectory.x[0], trajectory.y[0], s=45, marker="o", color=color,
            edgecolors="white", linewidths=0.8, zorder=20
        )
        axis.scatter(
            trajectory.x[-1], trajectory.y[-1], s=55, marker="x", color=color,
            linewidths=2.0, zorder=20
        )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(title)
    axis.grid(True, color="white", linewidth=0.4, alpha=0.25)
    axis.legend(loc="lower right", framealpha=0.92)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--factor-graph", type=Path, required=True)
    parser.add_argument(
        "--trajectory",
        action="append",
        default=[],
        type=parse_trajectory_specification,
        metavar="LABEL=CSV",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--title",
        default="Comparação de trajetórias",
        help="Title displayed above the plot",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_trajectory(args.factor_graph, "Factor Graph 2D")
    aligned: list[Trajectory] = [reference]
    input_files: dict[str, str] = {reference.label: str(args.factor_graph.resolve())}
    for label, path in args.trajectory:
        trajectory = load_trajectory(path, label)
        aligned.append(align_initial_pose(trajectory, reference))
        input_files[label] = str(path.resolve())

    aligned_csv = args.output_dir / "trajectories_aligned.csv"
    metrics_json = args.output_dir / "trajectory_comparison_metrics.json"
    plot_png = args.output_dir / "trajectory_comparison_all_methods.png"
    write_aligned_csv(aligned_csv, aligned)
    metrics = {
        "alignment": (
            "SE(2) rígido pela primeira pose de cada método para a pose do Factor "
            "Graph interpolada no mesmo timestamp; sem escala e sem ajuste global"
        ),
        "metric_sampling": (
            "Comprimento e erros calculados depois de interpolar cada método nos "
            "timestamps do Factor Graph; raw_path_length_m mantém o comprimento "
            "direto à frequência original de publicação"
        ),
        "map_yaml": str(args.map_yaml.resolve()),
        "inputs": input_files,
        "methods": {
            trajectory.label: trajectory_metrics(trajectory, reference)
            for trajectory in aligned
        },
    }
    with metrics_json.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    make_plot(plot_png, args.map_yaml, aligned, args.dpi, args.title)
    print(f"Trajetórias alinhadas: {aligned_csv}")
    print(f"Métricas: {metrics_json}")
    print(f"Figura: {plot_png}")


if __name__ == "__main__":
    main()
