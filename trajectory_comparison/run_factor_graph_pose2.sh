#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_factor_graph_pose2.sh BAG MAP_YAML OUTPUT_DIR \
    --initial-pose X Y YAW_DEG [options]

Options:
  --config YAML       Factor Graph profile (default: configs/hospital.yaml)
  --start-offset SEC  Bag offset (default: 90)
  --duration SEC      0 means to the end (default: 0)
  --rate RATE         Playback rate (default: 0.5)
  --domain-id ID      ROS domain (default: 97)
  --min-height M      Laser slice lower bound in base_link (default: 0.60)
  --max-height M      Laser slice upper bound in base_link (default: 0.80)
  --sensor-z M        base_link -> os_sensor height (default: 0.70)
  --overwrite         Replace existing Factor Graph CSV outputs

The bag supplies PointCloud2, filtered odometry and the Ouster IMU. This script
publishes the sensor TFs, converts the cloud to /scan, runs the Humble/GTSAM
Pose2 container, and replays only the required bag topics.
EOF
}

if (( $# < 3 )); then usage; exit 2; fi
bag=$1; map_yaml=$2; output_dir=$3; shift 3

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
config="$repo_root/configs/hospital.yaml"
start_offset=90
duration=0
rate=0.5
domain_id=97
min_height=0.60
max_height=0.80
sensor_z=0.70
overwrite=false
initial_x=""; initial_y=""; initial_yaw_deg=""

while (( $# )); do
  case "$1" in
    --config) config=$2; shift 2 ;;
    --start-offset) start_offset=$2; shift 2 ;;
    --duration) duration=$2; shift 2 ;;
    --rate) rate=$2; shift 2 ;;
    --domain-id) domain_id=$2; shift 2 ;;
    --min-height) min_height=$2; shift 2 ;;
    --max-height) max_height=$2; shift 2 ;;
    --sensor-z) sensor_z=$2; shift 2 ;;
    --initial-pose)
      (( $# >= 4 )) || { echo "--initial-pose needs X Y YAW_DEG" >&2; exit 2; }
      initial_x=$2; initial_y=$3; initial_yaw_deg=$4; shift 4 ;;
    --overwrite) overwrite=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -d "$bag" && -f "$bag/metadata.yaml" ]] || {
  echo "Bag not found: $bag" >&2; exit 1; }
[[ -f "$map_yaml" ]] || { echo "Map YAML not found: $map_yaml" >&2; exit 1; }
[[ -f "$config" ]] || { echo "Config not found: $config" >&2; exit 1; }
[[ -n "$initial_x" && -n "$initial_y" && -n "$initial_yaw_deg" ]] || {
  echo "--initial-pose X Y YAW_DEG is required" >&2; exit 2; }
for command in ros2 docker setsid; do
  command -v "$command" >/dev/null || { echo "$command is required." >&2; exit 1; }
done

trajectory_csv="$output_dir/trajectory.csv"
if [[ -e "$trajectory_csv" && "$overwrite" != true ]]; then
  echo "Refusing to overwrite $trajectory_csv (use --overwrite)." >&2
  exit 1
fi
mkdir -p "$output_dir/logs"
if [[ "$overwrite" == true ]]; then
  rm -f -- "$output_dir/trajectory.csv" "$output_dir/odometry_only.csv" \
    "$output_dir/trajectory_components.csv" "$output_dir/status.json" \
    "$output_dir/run_summary.txt"
fi

container_name="indoor-factor-graph-pose2-${domain_id}"
if docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "Docker container already exists: $container_name" >&2
  exit 1
fi

export ROS_DOMAIN_ID=$domain_id
export ROS_LOCALHOST_ONLY=0
unset SNAP SNAP_COMMON SNAP_DATA SNAP_USER_COMMON SNAP_USER_DATA SNAP_NAME \
  SNAP_INSTANCE_NAME SNAP_REVISION SNAP_VERSION

pids=()
bag_pid=""
factor_graph_pid=""
container_started=false

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
  if [[ "$container_started" == true ]]; then
    docker stop --time 10 "$container_name" >/dev/null 2>&1 || true
  fi
  stop_group "$factor_graph_pid"
  for (( index=${#pids[@]}-1; index>=0; index-- )); do
    stop_group "${pids[$index]}"
  done
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

setsid "$repo_root/pipeline.sh" run \
  --map "$map_yaml" --output "$output_dir" --config "$config" \
  --initial-pose "$initial_x" "$initial_y" "$initial_yaw_deg" \
  --domain-id "$domain_id" --name "$container_name" \
  -- --allow-missing-robot-imu \
  >"$output_dir/logs/factor_graph.log" 2>&1 &
factor_graph_pid=$!
container_started=true
sleep 10
docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null | grep -qx true || {
  echo "Factor Graph container did not start; inspect factor_graph.log" >&2; exit 1; }

bag_command=(
  ros2 bag play "$bag" --clock 40 --rate "$rate" --delay 2
  --start-offset "$start_offset" --disable-keyboard-controls
)
if [[ "$duration" != 0 && "$duration" != 0.0 ]]; then
  bag_command+=(--playback-duration "$duration")
fi
bag_command+=(--topics /odometry/filtered /ouster/imu /ouster/points)
printf '%q ' "${bag_command[@]}" >"$output_dir/bag_command.txt"
printf '\n' >>"$output_dir/bag_command.txt"
setsid "${bag_command[@]}" >"$output_dir/logs/bag_play.log" 2>&1 &
bag_pid=$!
set +e; wait "$bag_pid"; bag_status=$?; set -e
bag_pid=""
(( bag_status == 0 )) || { echo "Bag playback failed: $bag_status" >&2; exit "$bag_status"; }
sleep 10

docker stop --time 10 "$container_name" >/dev/null
container_started=false
wait "$factor_graph_pid" 2>/dev/null || true
factor_graph_pid=""

[[ -s "$trajectory_csv" ]] || {
  echo "Factor Graph did not create $trajectory_csv" >&2; exit 1; }
pose_count=$(($(wc -l <"$trajectory_csv") - 1))
(( pose_count > 0 )) || { echo "No Factor Graph poses recorded." >&2; exit 1; }
printf 'method=factor_graph_pose2\nbag=%s\nmap_yaml=%s\nstart_offset_s=%s\nduration_s=%s\nplayback_rate=%s\nposes=%s\ntrajectory=%s\n' \
  "$bag" "$map_yaml" "$start_offset" "$duration" "$rate" "$pose_count" \
  "$trajectory_csv" >"$output_dir/run_summary.txt"
echo "[Factor Graph 2D] $pose_count poses saved in $trajectory_csv"
