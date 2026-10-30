#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_localization_method.sh METHOD BAG MAP_YAML STATE_ROOT OUTPUT_DIR [options]

METHOD:
  amcl | slam_toolbox_localization | cartographer_localization |
  kiss_cartographer_localization | kiss_icp

By default, STATE_ROOT contains:
  slam_toolbox/urgency_reference.posegraph + .data
  cartographer/urgency_reference.pbstream

Use --slam-state-base and --cartographer-state for another dataset. When both
overrides are supplied, STATE_ROOT is only a required positional placeholder.

Options:
  --start-offset SEC       Test-bag offset (default: 90)
  --duration SEC           0 means to the end (default: 0)
  --rate RATE              Playback rate (default: 0.5)
  --domain-id ID           ROS domain (default: 92)
  --initial-x M            AMCL initial x (default: 5.330157484)
  --initial-y M            AMCL initial y (default: 45.834921258)
  --initial-yaw-deg DEG    AMCL initial yaw (default: -96.878981676)
  --slam-state-base PATH   Override SLAM Toolbox state basename (no extension)
  --cartographer-state PATH
                           Override the frozen Cartographer .pbstream
  --slam-start-x M         Start pose in the SLAM Toolbox native map (default: 0)
  --slam-start-y M         Start pose in the SLAM Toolbox native map (default: 0)
  --slam-start-yaw-deg DEG Start yaw in its native map (default: 0)
  --cartographer-start-x M
  --cartographer-start-y M
  --cartographer-start-yaw-deg DEG
                           Optional pose in the frozen pbstream map
  --min-height M           Laser slice lower bound (default: 0.60)
  --max-height M           Laser slice upper bound (default: 0.80)
  --sensor-z M             base_link -> os_sensor height (default: 0.70)
  --overwrite              Replace trajectory.csv
EOF
}

if (( $# < 5 )); then usage; exit 2; fi
method=$1; bag=$2; map_yaml=$3; state_root=$4; output_dir=$5
shift 5

start_offset=90
duration=0
rate=0.5
domain_id=92
initial_x=5.330157484
initial_y=45.834921258
initial_yaw_deg=-96.878981676
slam_state_override=""
cartographer_state_override=""
slam_start_x=0.0
slam_start_y=0.0
slam_start_yaw_deg=0.0
cartographer_start_x=""
cartographer_start_y=""
cartographer_start_yaw_deg=""
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
    --initial-x) initial_x=$2; shift 2 ;;
    --initial-y) initial_y=$2; shift 2 ;;
    --initial-yaw-deg) initial_yaw_deg=$2; shift 2 ;;
    --slam-state-base) slam_state_override=$2; shift 2 ;;
    --cartographer-state) cartographer_state_override=$2; shift 2 ;;
    --slam-start-x) slam_start_x=$2; shift 2 ;;
    --slam-start-y) slam_start_y=$2; shift 2 ;;
    --slam-start-yaw-deg) slam_start_yaw_deg=$2; shift 2 ;;
    --cartographer-start-x) cartographer_start_x=$2; shift 2 ;;
    --cartographer-start-y) cartographer_start_y=$2; shift 2 ;;
    --cartographer-start-yaw-deg) cartographer_start_yaw_deg=$2; shift 2 ;;
    --min-height) min_height=$2; shift 2 ;;
    --max-height) max_height=$2; shift 2 ;;
    --sensor-z) sensor_z=$2; shift 2 ;;
    --overwrite) overwrite=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$method" in
  amcl|slam_toolbox_localization|cartographer_localization|kiss_cartographer_localization|kiss_icp) ;;
  *) usage >&2; exit 2 ;;
esac
[[ -d "$bag" ]] || { echo "Bag not found: $bag" >&2; exit 1; }
[[ -f "$map_yaml" ]] || { echo "Map YAML not found: $map_yaml" >&2; exit 1; }
command -v ros2 >/dev/null || { echo "ROS 2 is not sourced." >&2; exit 1; }
cartographer_pose_values=(
  "$cartographer_start_x" "$cartographer_start_y" "$cartographer_start_yaw_deg"
)
cartographer_pose_count=0
for value in "${cartographer_pose_values[@]}"; do
  [[ -z "$value" ]] || ((cartographer_pose_count += 1))
done
(( cartographer_pose_count == 0 || cartographer_pose_count == 3 )) || {
  echo "Set all three Cartographer start-pose values, or none." >&2; exit 2; }

slam_state="$state_root/slam_toolbox/urgency_reference"
cartographer_state="$state_root/cartographer/urgency_reference.pbstream"
[[ -z "$slam_state_override" ]] || slam_state=$slam_state_override
[[ -z "$cartographer_state_override" ]] || cartographer_state=$cartographer_state_override
if [[ "$method" == slam_toolbox_localization ]]; then
  [[ -s "$slam_state.posegraph" && -s "$slam_state.data" ]] || {
    echo "Missing SLAM Toolbox state under $slam_state" >&2; exit 1; }
fi
if [[ "$method" == cartographer_localization || "$method" == kiss_cartographer_localization ]]; then
  [[ -s "$cartographer_state" ]] || {
    echo "Missing Cartographer state: $cartographer_state" >&2; exit 1; }
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
trajectory_csv="$output_dir/trajectory.csv"
if [[ -e "$trajectory_csv" && "$overwrite" != true ]]; then
  echo "Refusing to overwrite $trajectory_csv (use --overwrite)." >&2; exit 1
fi
mkdir -p "$output_dir/logs"
export ROS_DOMAIN_ID=$domain_id
export ROS_LOCALHOST_ONLY=0
# Avoid making SLAM Toolbox reinterpret absolute paths when this script is run
# from the integrated terminal of the Snap build of VS Code.
unset SNAP SNAP_COMMON SNAP_DATA SNAP_USER_COMMON SNAP_USER_DATA SNAP_NAME \
  SNAP_INSTANCE_NAME SNAP_REVISION SNAP_VERSION

pids=()
bag_pid=""
recorder_pid=""
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
  stop_group "$recorder_pid"
  for (( i=${#pids[@]}-1; i>=0; i-- )); do stop_group "${pids[$i]}"; done
}
trap cleanup EXIT INT TERM

start_process base_to_sensor ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z "$sensor_z" --roll 0 --pitch 0 --yaw 0 \
  --frame-id base_link --child-frame-id os_sensor --ros-args -p use_sim_time:=true
start_process sensor_to_lidar ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 0.03618 --roll 0 --pitch 0 --yaw 3.141592653589793 \
  --frame-id os_sensor --child-frame-id os_lidar --ros-args -p use_sim_time:=true

start_scan_converter() {
  local scan_topic=$1
  start_process pointcloud_to_laserscan \
    ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node --ros-args \
      -r cloud_in:=/ouster/points -r scan:="$scan_topic" \
      -p use_sim_time:=true -p target_frame:=base_link \
      -p transform_tolerance:=0.10 -p min_height:="$min_height" -p max_height:="$max_height" \
      -p angle_min:=-3.141592653589793 -p angle_max:=3.141592653589793 \
      -p angle_increment:=0.004363323 -p scan_time:=0.1 \
      -p range_min:=0.30 -p range_max:=40.0 -p use_inf:=true \
      -p inf_epsilon:=1.0 -p queue_size:=100
}
start_kiss() {
  start_process kiss_icp ros2 run kiss_icp kiss_icp_node --ros-args \
    -r pointcloud_topic:=/ouster/points -r kiss/odometry:=/kiss/odometry \
    -p use_sim_time:=true -p base_frame:=base_link -p lidar_odom_frame:=kiss_odom \
    -p publish_odom_tf:=false -p invert_odom_tf:=false -p publish_debug_clouds:=false \
    -p data.deskew:=true -p data.min_range:=0.3 -p data.max_range:=60.0 \
    -p mapping.voxel_size:=0.25 -p adaptive_threshold.initial_threshold:=2.0 \
    -p adaptive_threshold.min_motion_th:=0.1 -p registration.max_num_iterations:=500 \
    -p registration.convergence_criterion:=0.0001 -p registration.max_num_threads:=0
}
activate_lifecycle() {
  local node=$1; local log_name=$2
  for _ in {1..40}; do
    if timeout 2 ros2 lifecycle get "$node" >/dev/null 2>&1; then break; fi
    sleep 0.25
  done
  timeout 120 ros2 lifecycle set "$node" configure >>"$output_dir/logs/$log_name.log" 2>&1
  timeout 120 ros2 lifecycle set "$node" activate >>"$output_dir/logs/$log_name.log" 2>&1
}
start_cartographer_trajectory() {
  [[ -n "$cartographer_start_x" ]] || return 0
  local quaternion
  quaternion=$(python3 -c \
    'import math, sys; a=math.radians(float(sys.argv[1]))/2; print(math.sin(a), math.cos(a))' \
    "$cartographer_start_yaw_deg")
  local qz=${quaternion%% *}
  local qw=${quaternion##* }
  timeout 120 ros2 service call /start_trajectory \
    cartographer_ros_msgs/srv/StartTrajectory \
    "{configuration_directory: '$script_dir/config', configuration_basename: 'cartographer_2d_pure_localization.lua', use_initial_pose: true, initial_pose: {position: {x: $cartographer_start_x, y: $cartographer_start_y, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: $qz, w: $qw}}, relative_to_trajectory_id: 0}" \
    >"$output_dir/logs/start_trajectory.log" 2>&1
}

recorder_mode=tf
recorder_source=""
recorder_extra=(
  --world-frame map --child-frame base_link --rate 20.0
  --gate-topic /scan --gate-delay 0.5
)
bag_topics=(/ouster/points)

case "$method" in
  amcl)
    recorder_source="AMCL"
    bag_topics+=(/tf)
    start_scan_converter /scan
    initial_yaw_rad=$(python3 -c \
      'import math, sys; print(math.radians(float(sys.argv[1])))' "$initial_yaw_deg")
    start_process map_server ros2 run nav2_map_server map_server --ros-args \
      -p use_sim_time:=true -p yaml_filename:="$map_yaml" \
      -p topic_name:=map -p frame_id:=map
    sleep 1
    activate_lifecycle /map_server lifecycle_map_server
    start_process amcl ros2 run nav2_amcl amcl --ros-args \
      --params-file "$script_dir/config/amcl_localization.yaml" \
      -p initial_pose.x:="$initial_x" -p initial_pose.y:="$initial_y" \
      -p initial_pose.z:=0.0 -p initial_pose.yaw:="$initial_yaw_rad"
    sleep 1
    activate_lifecycle /amcl lifecycle_amcl
    ;;
  slam_toolbox_localization)
    recorder_source="SLAM Toolbox Localization"
    bag_topics+=(/tf)
    start_scan_converter /scan
    slam_start_yaw_rad=$(python3 -c \
      'import math, sys; print(math.radians(float(sys.argv[1])))' "$slam_start_yaw_deg")
    start_process slam_toolbox ros2 run slam_toolbox localization_slam_toolbox_node --ros-args \
      --params-file "$script_dir/config/slam_toolbox_localization.yaml" \
      -p map_file_name:="$slam_state" \
      -p map_start_pose:="[$slam_start_x, $slam_start_y, $slam_start_yaw_rad]"
    sleep 2
    activate_lifecycle /slam_toolbox lifecycle_slam_toolbox
    ;;
  cartographer_localization)
    recorder_source="Cartographer Pure Localization"
    bag_topics+=(/odometry/filtered)
    start_scan_converter /scan
    cartographer_start_args=()
    [[ -z "$cartographer_start_x" ]] || \
      cartographer_start_args=(-start_trajectory_with_default_topics=false)
    start_process cartographer ros2 run cartographer_ros cartographer_node \
      -configuration_directory "$script_dir/config" \
      -configuration_basename cartographer_2d_pure_localization.lua \
      -load_state_filename "$cartographer_state" -load_frozen_state=true \
      "${cartographer_start_args[@]}" \
      --ros-args -p use_sim_time:=true -r scan:=/scan -r odom:=/odometry/filtered
    sleep 5
    start_cartographer_trajectory
    ;;
  kiss_cartographer_localization)
    recorder_source="KISS-ICP + Cartographer Pure Localization"
    start_scan_converter /scan_raw
    start_kiss
    start_process scan_restamper ros2 run kiss_cartographer_mapping scan_restamper.py --ros-args \
      -p use_sim_time:=true -p input_topic:=/scan_raw -p output_topic:=/scan \
      -p restamp_with_current_time:=true -p zero_time_fields:=true
    start_process odometry_restamper ros2 run kiss_cartographer_mapping odometry_restamper.py --ros-args \
      -p use_sim_time:=true -p input_topic:=/kiss/odometry \
      -p output_topic:=/kiss/odometry_restamped -p restamp_with_current_time:=true
    cartographer_start_args=()
    [[ -z "$cartographer_start_x" ]] || \
      cartographer_start_args=(-start_trajectory_with_default_topics=false)
    start_process cartographer ros2 run cartographer_ros cartographer_node \
      -configuration_directory "$script_dir/config" \
      -configuration_basename cartographer_2d_pure_localization.lua \
      -load_state_filename "$cartographer_state" -load_frozen_state=true \
      "${cartographer_start_args[@]}" \
      --ros-args -p use_sim_time:=true -r scan:=/scan -r odom:=/kiss/odometry_restamped
    sleep 5
    start_cartographer_trajectory
    ;;
  kiss_icp)
    recorder_mode=odometry
    recorder_source="KISS-ICP Odometry"
    recorder_extra=(--topic /kiss/odometry)
    start_kiss
    ;;
esac

setsid python3 "$script_dir/record_trajectory.py" \
  --mode "$recorder_mode" --source "$recorder_source" --output "$trajectory_csv" \
  "${recorder_extra[@]}" --ros-args -p use_sim_time:=true \
  >"$output_dir/logs/recorder.log" 2>&1 &
recorder_pid=$!
sleep 3

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
sleep 8
stop_group "$recorder_pid"; recorder_pid=""
(( bag_status == 0 )) || { echo "Bag playback failed: $bag_status" >&2; exit "$bag_status"; }
[[ -f "$trajectory_csv" ]] || { echo "Recorder did not create $trajectory_csv" >&2; exit 1; }
pose_count=$(($(wc -l <"$trajectory_csv") - 1))
(( pose_count > 0 )) || { echo "No poses recorded; inspect $output_dir/logs" >&2; exit 1; }

cat >"$output_dir/run_summary.txt" <<EOF
method=$method
source=$recorder_source
bag=$bag
map_yaml=$map_yaml
state_root=$state_root
start_offset_s=$start_offset
duration_s=$duration
playback_rate=$rate
initial_x=$initial_x
initial_y=$initial_y
initial_yaw_deg=$initial_yaw_deg
slam_state=$slam_state
slam_start_x=$slam_start_x
slam_start_y=$slam_start_y
slam_start_yaw_deg=$slam_start_yaw_deg
cartographer_state=$cartographer_state
cartographer_start_x=$cartographer_start_x
cartographer_start_y=$cartographer_start_y
cartographer_start_yaw_deg=$cartographer_start_yaw_deg
poses=$pose_count
trajectory=$trajectory_csv
EOF
echo "[$recorder_source] $pose_count poses saved in $trajectory_csv"
