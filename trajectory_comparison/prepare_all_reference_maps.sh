#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: prepare_all_reference_maps.sh REFERENCE_BAG OUTPUT_ROOT [options]" >&2
  exit 2
fi
bag=$1; output_root=$2; shift 2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$output_root"
"$script_dir/prepare_reference_map.sh" slam_toolbox "$bag" "$output_root/slam_toolbox" \
  --domain-id 90 "$@"
"$script_dir/prepare_reference_map.sh" cartographer "$bag" "$output_root/cartographer" \
  --domain-id 91 "$@"
echo "Reference states ready under $output_root"
