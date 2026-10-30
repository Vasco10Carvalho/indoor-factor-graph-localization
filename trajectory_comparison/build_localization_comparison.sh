#!/usr/bin/env bash
set -euo pipefail
if (( $# < 4 || $# > 5 )); then
  echo "Usage: build_localization_comparison.sh MAP_YAML FACTOR_GRAPH_CSV RAW_RUNS_DIR OUTPUT_DIR [TITLE]" >&2
  exit 2
fi
map_yaml=$1; factor_graph_csv=$2; raw_runs=$3; output_dir=$4
title=${5:-"Comparação de trajetórias"}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 "$script_dir/compare_trajectories.py" \
  --map-yaml "$map_yaml" --factor-graph "$factor_graph_csv" \
  --trajectory "AMCL=$raw_runs/amcl/trajectory.csv" \
  --trajectory "SLAM Toolbox Localization=$raw_runs/slam_toolbox_localization/trajectory.csv" \
  --trajectory "Cartographer Pure Localization=$raw_runs/cartographer_localization/trajectory.csv" \
  --trajectory "KISS-ICP + Cartographer Pure Localization=$raw_runs/kiss_cartographer_localization/trajectory.csv" \
  --trajectory "KISS-ICP Odometry=$raw_runs/kiss_icp/trajectory.csv" \
  --output-dir "$output_dir" --title "$title"
