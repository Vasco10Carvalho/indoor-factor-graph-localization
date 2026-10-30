#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  echo "Usage: build_comparison.sh MAP_YAML FACTOR_GRAPH_CSV RAW_RUNS_DIR OUTPUT_DIR" >&2
  exit 2
fi

map_yaml=$1
factor_graph_csv=$2
raw_runs=$3
output_dir=$4
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python3 "$script_dir/compare_trajectories.py" \
  --map-yaml "$map_yaml" \
  --factor-graph "$factor_graph_csv" \
  --trajectory "SLAM Toolbox=$raw_runs/slam_toolbox/trajectory.csv" \
  --trajectory "KISS-ICP=$raw_runs/kiss_icp/trajectory.csv" \
  --trajectory "Cartographer=$raw_runs/cartographer/trajectory.csv" \
  --trajectory "KISS-ICP + Cartographer=$raw_runs/kiss_cartographer/trajectory.csv" \
  --output-dir "$output_dir"
