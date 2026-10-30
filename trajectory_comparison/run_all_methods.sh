#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: run_all_methods.sh BAG OUTPUT_ROOT [run_method.sh options except --domain-id]" >&2
  exit 2
fi

bag=$1
output_root=$2
shift 2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$output_root"

methods=(slam_toolbox kiss_icp cartographer kiss_cartographer)
domain_ids=(84 85 86 87)
for index in "${!methods[@]}"; do
  method=${methods[$index]}
  echo "=== Running $method ==="
  "$script_dir/run_method.sh" "$method" "$bag" "$output_root/$method" \
    --domain-id "${domain_ids[$index]}" "$@"
done

echo "All raw trajectories are under $output_root"
