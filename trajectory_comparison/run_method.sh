#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_method.sh METHOD BAG OUTPUT_DIR [options]

METHOD: slam_toolbox | kiss_icp | cartographer | kiss_cartographer

Options:
  --start-offset SEC     Bag offset (default: 90)
  --duration SEC         Process only SEC seconds; 0 means until the end (default: 0)
  --rate RATE            Bag playback rate (default: 1.0)
  --domain-id ID         ROS domain for this run (default: 84)
  --min-height M         LiDAR slice lower limit in base_link (default: 0.60)
  --max-height M         LiDAR slice upper limit in base_link (default: 0.80)
  --sensor-z M           base_link -> os_sensor height (default: 0.70)
  --overwrite            Replace an existing trajectory.csv
EOF
}

if (( $# < 3 )); then
  usage
  exit 2
fi

method=$1
bag=$2
output_dir=$3
shift 3

start_offset=90
duration=0
rate=1.0
domain_id=84
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
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$method" in
  slam_toolbox|kiss_icp|cartographer|kiss_cartographer) ;;
  *) echo "Unknown method: $method" >&2; usage >&2; exit 2 ;;
esac

if [[ ! -d "$bag" ]]; then
  echo "Bag directory does not exist: $bag" >&2
  exit 1
fi
if ! command -v ros2 >/dev/null 2>&1; then
  echo "ROS 2 is not sourced. Source /opt/ros/jazzy/setup.bash and the KISS workspace." >&2
  exit 1
fi
for package in tf2_ros pointcloud_to_laserscan; do
  ros2 pkg prefix "$package" >/dev/null
done
if [[ "$method" == kiss_icp || "$method" == kiss_cartographer ]]; then
  ros2 pkg prefix kiss_icp >/dev/null
fi
if [[ "$method" == cartographer || "$method" == kiss_cartographer ]]; then
  ros2 pkg prefix cartographer_ros >/dev/null
fi
if [[ "$method" == slam_toolbox ]]; then
  ros2 pkg prefix slam_toolbox >/dev/null
fi
if [[ "$method" == kiss_cartographer ]]; then
  ros2 pkg prefix kiss_cartographer_mapping >/dev/null
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
trajectory_csv="$output_dir/trajectory.csv"
if [[ -e "$trajectory_csv" && "$overwrite" != true ]]; then
  echo "Refusing to overwrite $trajectory_csv (use --overwrite)." >&2
  exit 1
fi
mkdir -p "$output_dir/logs"

export ROS_DOMAIN_ID=$domain_id
export ROS_LOCALHOST_ONLY=0

pids=()
recorder_pid=""
bag_pid=""

start_process() {
  local log_name=$1
  shift
  setsid "$@" >"$output_dir/logs/$log_name.log" 2>&1 &
  pids+=("$!")
}

stop_pid() {
  local pid=${1:-}
  [[ -z "$pid" ]] && return 0
  # Every managed command starts in its own session. Signalling the process
  # group also stops children created by `ros2 run` and `ros2 launch`.
  kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
  for _ in {1..30}; do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..20}; do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  stop_pid "$bag_pid"
  stop_pid "$recorder_pid"
  local index
  for (( index=${#pids[@]}-1; index>=0; index-- )); do
    stop_pid "${pids[$index]}"
  done
}
trap cleanup EXIT INT TERM

start_process base_to_sensor \
  ros2 run tf2_ros static_transform_publisher \
    --x 0 --y 0 --z "$sensor_z" --roll 0 --pitch 0 --yaw 0 \
    --frame-id base_link --child-frame-id os_sensor \
    --ros-args -p use_sim_time:=true
start_process sensor_to_lidar \
  ros2 run tf2_ros static_transform_publisher \
    --x 0 --y 0 --z 0.03618 --roll 0 --pitch 0 --yaw 3.141592653589793 \
    --frame-id os_sensor --child-frame-id os_lidar \
    --ros-args -p use_sim_time:=true

start_scan_converter() {
  local output_topic=$1
  start_process pointcloud_to_laserscan \
    ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node --ros-args \
      -r cloud_in:=/ouster/points -r scan:="$output_topic" \
      -p use_sim_time:=true -p target_frame:=base_link \
      -p transform_tolerance:=0.10 \
      -p min_height:="$min_height" -p max_height:="$max_height" \
      -p angle_min:=-3.141592653589793 -p angle_max:=3.141592653589793 \
      -p angle_increment:=0.004363323 -p scan_time:=0.1 \
      -p range_min:=0.30 -p range_max:=40.0 \
      -p use_inf:=true -p inf_epsilon:=1.0 -p queue_size:=100
}

start_kiss() {
  start_process kiss_icp \
    ros2 run kiss_icp kiss_icp_node --ros-args \
      -r pointcloud_topic:=/ouster/points -r kiss/odometry:=/kiss/odometry \
      -p use_sim_time:=true -p base_frame:=base_link \
      -p lidar_odom_frame:=kiss_odom -p publish_odom_tf:=false \
      -p invert_odom_tf:=false -p publish_debug_clouds:=false \
      -p data.deskew:=true -p data.min_range:=0.3 -p data.max_range:=60.0 \
      -p mapping.voxel_size:=0.25 \
      -p adaptive_threshold.initial_threshold:=2.0 \
      -p adaptive_threshold.min_motion_th:=0.1 \
      -p registration.max_num_iterations:=500 \
      -p registration.convergence_criterion:=0.0001 \
      -p registration.max_num_threads:=0
}

start_cartographer() {
  local odom_topic=$1
  start_process cartographer \
    ros2 run cartographer_ros cartographer_node \
      -configuration_directory "$script_dir/config" \
      -configuration_basename cartographer_2d_comparison.lua \
      --ros-args -p use_sim_time:=true -r scan:=/scan -r odom:="$odom_topic"
}

recorder_mode=tf
recorder_source=""
recorder_extra=(--world-frame map --child-frame base_link --rate 20.0)
bag_topics=(/ouster/points)

case "$method" in
  slam_toolbox)
    recorder_source="SLAM Toolbox"
    bag_topics+=(/tf)
    start_scan_converter /scan
    start_process slam_toolbox \
      ros2 launch slam_toolbox online_sync_launch.py \
        slam_params_file:="$script_dir/config/slam_toolbox_comparison.yaml" \
        use_sim_time:=true autostart:=true use_lifecycle_manager:=false
    ;;
  kiss_icp)
    recorder_mode=odometry
    recorder_source="KISS-ICP"
    recorder_extra=(--topic /kiss/odometry)
    start_kiss
    ;;
  cartographer)
    recorder_source="Cartographer"
    bag_topics+=(/odometry/filtered)
    start_scan_converter /scan
    start_cartographer /odometry/filtered
    ;;
  kiss_cartographer)
    recorder_source="KISS-ICP + Cartographer"
    start_scan_converter /scan_raw
    start_kiss
    start_process scan_restamper \
      ros2 run kiss_cartographer_mapping scan_restamper.py --ros-args \
        -p use_sim_time:=true -p input_topic:=/scan_raw -p output_topic:=/scan \
        -p restamp_with_current_time:=true -p zero_time_fields:=true
    start_process odometry_restamper \
      ros2 run kiss_cartographer_mapping odometry_restamper.py --ros-args \
        -p use_sim_time:=true -p input_topic:=/kiss/odometry \
        -p output_topic:=/kiss/odometry_restamped \
        -p restamp_with_current_time:=true
    start_cartographer /kiss/odometry_restamped
    ;;
esac

setsid python3 "$script_dir/record_trajectory.py" \
  --mode "$recorder_mode" --source "$recorder_source" \
  --output "$trajectory_csv" "${recorder_extra[@]}" \
  --ros-args -p use_sim_time:=true \
  >"$output_dir/logs/recorder.log" 2>&1 &
recorder_pid=$!

sleep 4
bag_command=(ros2 bag play "$bag" --clock 40 --rate "$rate" --delay 2 --start-offset "$start_offset")
if [[ "$duration" != 0 && "$duration" != 0.0 ]]; then
  bag_command+=(--playback-duration "$duration")
fi
bag_command+=(--topics "${bag_topics[@]}")
printf '%q ' "${bag_command[@]}" >"$output_dir/bag_command.txt"
printf '\n' >>"$output_dir/bag_command.txt"
setsid "${bag_command[@]}" >"$output_dir/logs/bag_play.log" 2>&1 &
bag_pid=$!
set +e
wait "$bag_pid"
bag_status=$?
set -e
bag_pid=""
sleep 8
stop_pid "$recorder_pid"
recorder_pid=""

if [[ $bag_status -ne 0 ]]; then
  echo "Bag playback failed (status $bag_status). See $output_dir/logs/bag_play.log" >&2
  exit "$bag_status"
fi
if [[ ! -f "$trajectory_csv" ]]; then
  echo "Recorder did not create $trajectory_csv. Inspect $output_dir/logs/recorder.log." >&2
  exit 1
fi
pose_count=$(($(wc -l <"$trajectory_csv") - 1))
if (( pose_count <= 0 )); then
  echo "No poses were recorded. Inspect the logs in $output_dir/logs." >&2
  exit 1
fi

cat >"$output_dir/run_summary.txt" <<EOF
method=$method
source=$recorder_source
bag=$bag
start_offset_s=$start_offset
duration_s=$duration
playback_rate=$rate
domain_id=$domain_id
min_height_m=$min_height
max_height_m=$max_height
sensor_z_m=$sensor_z
poses=$pose_count
trajectory=$trajectory_csv
EOF

echo "[$recorder_source] $pose_count poses saved in $trajectory_csv"
