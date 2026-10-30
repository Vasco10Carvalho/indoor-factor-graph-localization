#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

args=("$@")
has_config=false
for arg in "${args[@]}"; do
  if [[ "$arg" == "--config" ]]; then
    has_config=true
    break
  fi
done
if [[ "$has_config" == false ]]; then
  args+=(--config "$repo_root/configs/pharmacy_pose2.yaml")
fi

exec "$script_dir/run_factor_graph_pose2.sh" "${args[@]}"
