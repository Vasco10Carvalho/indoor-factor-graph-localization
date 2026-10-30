#!/usr/bin/env bash
set -euo pipefail
if (( $# < 4 )); then
  echo "Usage: run_all_localization_methods.sh BAG MAP_YAML STATE_ROOT OUTPUT_ROOT [options]" >&2
  exit 2
fi
bag=$1; map_yaml=$2; state_root=$3; output_root=$4; shift 4
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$output_root"
methods=(amcl slam_toolbox_localization cartographer_localization kiss_cartographer_localization kiss_icp)
domains=(92 93 94 95 96)
for index in "${!methods[@]}"; do
  method=${methods[$index]}
  echo "=== Running $method ==="
  "$script_dir/run_localization_method.sh" "$method" "$bag" "$map_yaml" \
    "$state_root" "$output_root/$method" --domain-id "${domains[$index]}" "$@"
done
echo "All localization trajectories are under $output_root"
