#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  prepare_reference_map.sh METHOD REFERENCE_BAG OUTPUT_DIR [options]

METHOD: slam_toolbox | cartographer

Options:
  --start-offset SEC  Reference-bag offset (default: 0)
  --duration SEC      0 means to the end (default: 0)
  --rate RATE         Playback rate (default: 0.5)
  --domain-id ID      ROS domain (default: 90)
  --min-height M      Laser slice lower bound (default: 0.60)
  --max-height M      Laser slice upper bound (default: 0.80)
  --sensor-z M        base_link -> os_sensor height (default: 0.70)
  --overwrite         Replace a pre-existing state
EOF
}

if (( $# < 3 )); then usage; exit 2; fi
method=$1
bag=$2
output_dir=$3
shift 3

start_offset=0
duration=0
rate=0.5
domain_id=90
min_height=0.60
max_height=0.80
sensor_z=0.70
overwrite=false
while (( $# )); do
  case "$1" in
    --start-offset) start_offset=$2; shift 2 ;;
    --duration) duration=$2; shift 2 ;;
    --rate) rate=$2; shift 2 ;;
    --domain-id) domain_id=$2; shift 2 ;;
    --min-height) min_height=$2; shift 2 ;;
    --max-height) max_height=$2; shift 2 ;;
    --sensor-z) sensor_z=$2; shift 2 ;;
    --overwrite) overwrite=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$method" in slam_toolbox|cartographer) ;; *) usage >&2; exit 2 ;; esac
[[ -d "$bag" ]] || { echo "Bag not found: $bag" >&2; exit 1; }
command -v ros2 >/dev/null || { echo "ROS 2 is not sourced." >&2; exit 1; }
command -v setsid >/dev/null || { echo "setsid is required." >&2; exit 1; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$output_dir/logs"
state_base="$output_dir/urgency_reference"
if [[ "$method" == slam_toolbox ]]; then
  state_file="$state_base.posegraph"
else
  state_file="$state_base.pbstream"
fi
if [[ -e "$state_file" && "$overwrite" != true ]]; then
  echo "Refusing to overwrite $state_file (use --overwrite)." >&2
  exit 1
fi
if [[ "$overwrite" == true ]]; then
  rm -f -- "$state_base.posegraph" "$state_base.data" "$state_base.pbstream" \
    "$state_base.pgm" "$state_base.yaml"
fi

export ROS_DOMAIN_ID=$domain_id
export ROS_LOCALHOST_ONLY=0
# VS Code installed as a Snap exports SNAP_COMMON to every integrated shell.
# SLAM Toolbox interprets that as "running inside a snap" and prefixes absolute
# serialization paths with /var/snap/code/common, so posegraph creation fails.
unset SNAP SNAP_COMMON SNAP_DATA SNAP_USER_COMMON SNAP_USER_DATA SNAP_NAME \
  SNAP_INSTANCE_NAME SNAP_REVISION SNAP_VERSION
pids=()
bag_pid=""

start_process() {
  local name=$1; shift
  setsid "$@" >"$output_dir/logs/$name.log" 2>&1 &
  pids+=("$!")
}
stop_group() {
  local pid=${1:-}; [[ -n "$pid" ]] || return 0
  kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
  for _ in {1..50}; do kill -0 -- "-$pid" 2>/dev/null || break; sleep 0.1; done
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in {1..20}; do kill -0 -- "-$pid" 2>/dev/null || break; sleep 0.1; done
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  trap - EXIT INT TERM
  stop_group "$bag_pid"
  for (( i=${#pids[@]}-1; i>=0; i-- )); do stop_group "${pids[$i]}"; done
}
trap cleanup EXIT INT TERM

start_process base_to_sensor ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z "$sensor_z" --roll 0 --pitch 0 --yaw 0 \
  --frame-id base_link --child-frame-id os_sensor --ros-args -p use_sim_time:=true
start_process sensor_to_lidar ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 0.03618 --roll 0 --pitch 0 --yaw 3.141592653589793 \
  --frame-id os_sensor --child-frame-id os_lidar --ros-args -p use_sim_time:=true
start_process pointcloud_to_laserscan \
  ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node --ros-args \
    -r cloud_in:=/ouster/points -r scan:=/scan \
    -p use_sim_time:=true -p target_frame:=base_link \
    -p transform_tolerance:=0.10 -p min_height:="$min_height" -p max_height:="$max_height" \
    -p angle_min:=-3.141592653589793 -p angle_max:=3.141592653589793 \
    -p angle_increment:=0.004363323 -p scan_time:=0.1 \
    -p range_min:=0.30 -p range_max:=40.0 -p use_inf:=true \
    -p inf_epsilon:=1.0 -p queue_size:=100

bag_topics=(/ouster/points)
if [[ "$method" == slam_toolbox ]]; then
  bag_topics+=(/tf)
  start_process mapper ros2 launch slam_toolbox online_sync_launch.py \
    slam_params_file:="$script_dir/config/slam_toolbox_comparison.yaml" \
    use_sim_time:=true autostart:=true use_lifecycle_manager:=false
else
  bag_topics+=(/odometry/filtered)
  start_process mapper ros2 run cartographer_ros cartographer_node \
    -configuration_directory "$script_dir/config" \
    -configuration_basename cartographer_2d_comparison.lua \
    --ros-args -p use_sim_time:=true -r scan:=/scan -r odom:=/odometry/filtered
fi

sleep 5
bag_command=(ros2 bag play "$bag" --clock 40 --rate "$rate" --delay 2 --start-offset "$start_offset")
if [[ "$duration" != 0 && "$duration" != 0.0 ]]; then
  bag_command+=(--playback-duration "$duration")
fi
bag_command+=(--topics "${bag_topics[@]}")
printf '%q ' "${bag_command[@]}" >"$output_dir/bag_command.txt"; printf '\n' >>"$output_dir/bag_command.txt"
setsid "${bag_command[@]}" >"$output_dir/logs/bag_play.log" 2>&1 &
bag_pid=$!
set +e; wait "$bag_pid"; bag_status=$?; set -e
bag_pid=""
(( bag_status == 0 )) || { echo "Bag playback failed: $bag_status" >&2; exit "$bag_status"; }
sleep 15

if [[ "$method" == slam_toolbox ]]; then
  timeout 180 ros2 service call /slam_toolbox/serialize_map \
    slam_toolbox/srv/SerializePoseGraph "{filename: '$state_base'}" \
    >"$output_dir/logs/serialize_state.log" 2>&1
  timeout 60 ros2 service call /slam_toolbox/save_map \
    slam_toolbox/srv/SaveMap "{name: {data: '$state_base'}}" \
    >"$output_dir/logs/save_map.log" 2>&1 || true
  [[ -s "$state_base.posegraph" && -s "$state_base.data" ]] || {
    echo "SLAM Toolbox did not create the serialized state." >&2; exit 1; }
else
  timeout 180 ros2 service call /finish_trajectory \
    cartographer_ros_msgs/srv/FinishTrajectory '{trajectory_id: 0}' \
    >"$output_dir/logs/finish_trajectory.log" 2>&1
  timeout 180 ros2 service call /write_state \
    cartographer_ros_msgs/srv/WriteState \
    "{filename: '$state_base.pbstream', include_unfinished_submaps: true}" \
    >"$output_dir/logs/write_state.log" 2>&1
  [[ -s "$state_base.pbstream" ]] || {
    echo "Cartographer did not create the pbstream." >&2; exit 1; }
  ros2 run cartographer_ros cartographer_pbstream_to_ros_map \
    -pbstream_filename="$state_base.pbstream" \
    -map_filestem="$state_base" -resolution=0.05 \
    >"$output_dir/logs/pbstream_to_map.log" 2>&1
fi

printf 'method=%s\nreference_bag=%s\nstate=%s\n' "$method" "$bag" "$state_file" \
  >"$output_dir/reference_summary.txt"
echo "[$method] reference state saved: $state_file"
