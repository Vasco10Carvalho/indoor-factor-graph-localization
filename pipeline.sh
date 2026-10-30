#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${INDOOR_IMAGE_NAME:-indoor-factor-graph-localization:humble}"
DEFAULT_PROFILE="${ROOT_DIR}/configs/hospital.yaml"

usage() {
  cat <<'EOF'
Indoor factor-graph localization in ROS 2 Humble

Usage:
  ./pipeline.sh build
  ./pipeline.sh check
  ./pipeline.sh inspect --bag BAG_DIRECTORY
  ./pipeline.sh run --map MAP.yaml --output DIR --initial-pose X Y YAW_DEG [options]
  ./pipeline.sh run --map MAP.yaml --output DIR --wait-for-initial-pose [options]
  ./pipeline.sh play --bag BAG_DIRECTORY [options]
  ./pipeline.sh rviz [--domain-id ID] [--live]
  ./pipeline.sh monitor [--domain-id ID]
  ./pipeline.sh publish-result --output DIR --map-yaml MAP.yaml [--domain-id ID]
  ./pipeline.sh evaluate --output DIR [--map-yaml MAP.yaml] [--bag-label TEXT] [--map-label TEXT]

Run options:
  --config YAML              Algorithm profile (default: configs/hospital.yaml)
  --domain-id ID             ROS domain (default: 42)
  --live                     Use wall time instead of rosbag /clock
  --name NAME                Docker container name
  -- ARGS                    Extra factor-graph node arguments

Play options:
  --rate VALUE               Playback rate (default: 1.0)
  --start-offset SEC         Skip the beginning of the bag (default: 0.0)
  --storage ID               Optional storage plugin, e.g. sqlite3 or mcap
  --topics CSV               Optional comma-separated topic allow-list
  --domain-id ID             Must match the run/RViz domain
EOF
}

fail() {
  echo "[ERROR] $*" >&2
  exit 2
}

need_value() {
  [[ $# -ge 2 ]] || fail "Missing value after $1"
}

ros_environment() {
  local domain_id="$1"
  printf '%s\0' \
    --env "ROS_DOMAIN_ID=${domain_id}" \
    --env ROS_LOCALHOST_ONLY=0 \
    --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    --env FASTDDS_BUILTIN_TRANSPORTS=UDPv4
}

action="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${action}" in
  build)
    exec docker build --tag "${IMAGE_NAME}" "${ROOT_DIR}"
    ;;

  check)
    exec docker run --rm "${IMAGE_NAME}" bash -lc \
      "python3 -m py_compile /workspace/indoor_localization/*.py && \
       python3 -m unittest discover -s /workspace/tests -p 'test_*.py'"
    ;;

  inspect)
    bag=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --bag) need_value "$@"; bag="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown inspect option: $1" ;;
      esac
    done
    [[ -n "${bag}" ]] || fail "inspect requires --bag"
    [[ -f "${bag}/metadata.yaml" ]] || fail "metadata.yaml not found in ${bag}"
    bag="$(realpath "${bag}")"
    exec docker run --rm --volume "${bag}:/data/bag:ro" "${IMAGE_NAME}" \
      ros2 bag info /data/bag
    ;;

  run)
    map_yaml=""
    output=""
    profile="${DEFAULT_PROFILE}"
    domain_id=42
    use_sim_time=true
    container_name=indoor-factor-graph
    wait_for_pose=false
    initial_x=""
    initial_y=""
    initial_yaw=""
    extra_args=()

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --map) need_value "$@"; map_yaml="$2"; shift 2 ;;
        --output) need_value "$@"; output="$2"; shift 2 ;;
        --config) need_value "$@"; profile="$2"; shift 2 ;;
        --domain-id) need_value "$@"; domain_id="$2"; shift 2 ;;
        --name) need_value "$@"; container_name="$2"; shift 2 ;;
        --initial-pose)
          [[ $# -ge 4 ]] || fail "--initial-pose requires X Y YAW_DEG"
          initial_x="$2"; initial_y="$3"; initial_yaw="$4"; shift 4
          ;;
        --wait-for-initial-pose) wait_for_pose=true; shift ;;
        --live) use_sim_time=false; shift ;;
        --) shift; extra_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown run option: $1" ;;
      esac
    done

    [[ -f "${map_yaml}" ]] || fail "Map YAML not found: ${map_yaml:-<missing>}"
    [[ -f "${profile}" ]] || fail "Profile not found: ${profile}"
    [[ -n "${output}" ]] || fail "run requires --output"
    if [[ "${wait_for_pose}" != true && -z "${initial_x}" ]]; then
      fail "Use --initial-pose X Y YAW_DEG or --wait-for-initial-pose"
    fi
    if [[ "${wait_for_pose}" == true && -n "${initial_x}" ]]; then
      fail "Choose either --initial-pose or --wait-for-initial-pose"
    fi

    map_yaml="$(realpath "${map_yaml}")"
    profile="$(realpath "${profile}")"
    mkdir -p "${output}"
    output="$(realpath "${output}")"
    map_dir="$(dirname "${map_yaml}")"
    map_name="$(basename "${map_yaml}")"
    initial_args=()
    if [[ "${wait_for_pose}" == true ]]; then
      initial_args+=(--wait-for-initial-pose)
    else
      initial_args+=(--initial-x "${initial_x}" --initial-y "${initial_y}" --initial-yaw-deg "${initial_yaw}")
    fi
    mapfile -d '' ros_env < <(ros_environment "${domain_id}")

    exec docker run --rm --init \
      --name "${container_name}" \
      --network host --ipc host \
      --user "$(id -u):$(id -g)" \
      --env HOME=/tmp/indoor_localization_home \
      "${ros_env[@]}" \
      --volume "${map_dir}:/data/map:ro" \
      --volume "${profile}:/data/profile.yaml:ro" \
      --volume "${output}:/output" \
      "${IMAGE_NAME}" \
      python3 /workspace/indoor_localization/online_indoor_factor_graph_pose2.py \
        --config /data/profile.yaml \
        --map-yaml "/data/map/${map_name}" \
        --output-csv /output/trajectory.csv \
        --output-odom-only-csv /output/odometry_only.csv \
        --output-components-csv /output/trajectory_components.csv \
        --output-status-json /output/status.json \
        "${initial_args[@]}" \
        "${extra_args[@]}" \
        --ros-args -p use_sim_time:="${use_sim_time}"
    ;;

  play)
    bag=""
    domain_id=42
    rate=1.0
    start_offset=0.0
    storage=""
    topics=""
    delay=3
    container_name=indoor-bag-player
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --bag) need_value "$@"; bag="$2"; shift 2 ;;
        --domain-id) need_value "$@"; domain_id="$2"; shift 2 ;;
        --rate) need_value "$@"; rate="$2"; shift 2 ;;
        --start-offset) need_value "$@"; start_offset="$2"; shift 2 ;;
        --storage) need_value "$@"; storage="$2"; shift 2 ;;
        --topics) need_value "$@"; topics="$2"; shift 2 ;;
        --delay) need_value "$@"; delay="$2"; shift 2 ;;
        --name) need_value "$@"; container_name="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown play option: $1" ;;
      esac
    done
    [[ -f "${bag}/metadata.yaml" ]] || fail "metadata.yaml not found in ${bag:-<missing>}"
    bag="$(realpath "${bag}")"
    storage_args=()
    [[ -z "${storage}" ]] || storage_args+=(--storage "${storage}")
    topic_args=()
    if [[ -n "${topics}" ]]; then
      IFS=',' read -r -a topic_list <<< "${topics}"
      topic_args+=(--topics "${topic_list[@]}")
    fi
    mapfile -d '' ros_env < <(ros_environment "${domain_id}")

    exec docker run --rm --init \
      --name "${container_name}" --network host --ipc host \
      "${ros_env[@]}" \
      --volume "${bag}:/data/bag:ro" \
      "${IMAGE_NAME}" \
      ros2 bag play /data/bag \
        "${storage_args[@]}" \
        --clock 40 --rate "${rate}" --start-offset "${start_offset}" --delay "${delay}" \
        "${topic_args[@]}"
    ;;

  rviz)
    domain_id=42
    use_sim_time=true
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --domain-id) need_value "$@"; domain_id="$2"; shift 2 ;;
        --live) use_sim_time=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown rviz option: $1" ;;
      esac
    done
    [[ -n "${DISPLAY:-}" ]] || fail "DISPLAY is not set; RViz needs a graphical session"
    mapfile -d '' ros_env < <(ros_environment "${domain_id}")
    gui_args=(--env "DISPLAY=${DISPLAY}" --env QT_X11_NO_MITSHM=1 --volume /tmp/.X11-unix:/tmp/.X11-unix:rw)
    xauthority="${XAUTHORITY:-${HOME}/.Xauthority}"
    if [[ -f "${xauthority}" ]]; then
      gui_args+=(--env XAUTHORITY=/tmp/.docker.xauth --volume "${xauthority}:/tmp/.docker.xauth:ro")
    fi
    device_args=()
    [[ ! -e /dev/dri ]] || device_args+=(--device /dev/dri)
    exec docker run --rm --init --network host --ipc host \
      --user "$(id -u):$(id -g)" --env HOME=/tmp/rviz_home \
      "${ros_env[@]}" "${gui_args[@]}" "${device_args[@]}" \
      "${IMAGE_NAME}" rviz2 \
        -d /workspace/indoor_localization/indoor_localization.rviz \
        --ros-args -p use_sim_time:="${use_sim_time}"
    ;;

  monitor)
    domain_id=42
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --domain-id) need_value "$@"; domain_id="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown monitor option: $1" ;;
      esac
    done
    mapfile -d '' ros_env < <(ros_environment "${domain_id}")
    exec docker run --rm --init --network host "${ros_env[@]}" "${IMAGE_NAME}" \
      python3 /workspace/indoor_localization/monitor_status.py
    ;;

  publish-result)
    output=""
    map_yaml=""
    domain_id=42
    container_name=indoor-result-publisher
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --output) need_value "$@"; output="$2"; shift 2 ;;
        --map-yaml) need_value "$@"; map_yaml="$2"; shift 2 ;;
        --domain-id) need_value "$@"; domain_id="$2"; shift 2 ;;
        --name) need_value "$@"; container_name="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown publish-result option: $1" ;;
      esac
    done
    [[ -f "${output}/trajectory.csv" ]] || fail "trajectory.csv not found in ${output:-<missing>}"
    [[ -f "${output}/odometry_only.csv" ]] || fail "odometry_only.csv not found in ${output:-<missing>}"
    [[ -f "${output}/trajectory_components.csv" ]] || fail "trajectory_components.csv not found in ${output:-<missing>}"
    [[ -f "${map_yaml}" ]] || fail "Map YAML not found: ${map_yaml:-<missing>}"
    output="$(realpath "${output}")"
    map_yaml="$(realpath "${map_yaml}")"
    map_dir="$(dirname "${map_yaml}")"
    map_name="$(basename "${map_yaml}")"
    mapfile -d '' ros_env < <(ros_environment "${domain_id}")
    exec docker run --rm --init \
      --name "${container_name}" --network host --ipc host \
      --user "$(id -u):$(id -g)" --env HOME=/tmp/indoor_result_home \
      "${ros_env[@]}" \
      --volume "${output}:/output:ro" \
      --volume "${map_dir}:/data/map:ro" \
      "${IMAGE_NAME}" \
      python3 /workspace/indoor_localization/publish_indoor_result.py \
        --map-yaml "/data/map/${map_name}" \
        --trajectory-csv /output/trajectory.csv \
        --odom-csv /output/odometry_only.csv \
        --components-csv /output/trajectory_components.csv
    ;;

  evaluate)
    output=""
    map_yaml=""
    bag_label=""
    map_label=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --output) need_value "$@"; output="$2"; shift 2 ;;
        --map-yaml) need_value "$@"; map_yaml="$2"; shift 2 ;;
        --bag-label) need_value "$@"; bag_label="$2"; shift 2 ;;
        --map-label) need_value "$@"; map_label="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown evaluate option: $1" ;;
      esac
    done
    [[ -f "${output}/trajectory.csv" ]] || fail "trajectory.csv not found in ${output:-<missing>}"
    output="$(realpath "${output}")"
    optional_args=()
    [[ ! -f "${output}/status.json" ]] || optional_args+=(--status-json /output/status.json)
    [[ ! -f "${output}/odometry_only.csv" ]] || optional_args+=(--odom-csv /output/odometry_only.csv)
    if [[ -f "${output}/trajectory_components.csv" ]]; then
      optional_args+=(
        --components-csv /output/trajectory_components.csv
        --comparison-plot-out /output/trajectory_comparison.png
        --timeseries-plot-out /output/trajectory_timeseries.png
        --quality-plot-out /output/scan_quality.png
      )
    fi
    map_volume=()
    if [[ -n "${map_yaml}" ]]; then
      [[ -f "${map_yaml}" ]] || fail "Map YAML not found: ${map_yaml}"
      map_yaml="$(realpath "${map_yaml}")"
      map_dir="$(dirname "${map_yaml}")"
      map_name="$(basename "${map_yaml}")"
      map_volume+=(--volume "${map_dir}:/data/map:ro")
      optional_args+=(--map-yaml "/data/map/${map_name}")
    fi
    exec docker run --rm --init --user "$(id -u):$(id -g)" \
      --env HOME=/tmp/indoor_evaluation_home \
      --volume "${output}:/output" "${map_volume[@]}" "${IMAGE_NAME}" \
      python3 /workspace/indoor_localization/evaluate_indoor_result.py \
        /output/trajectory.csv "${optional_args[@]}" \
        --summary-out /output/summary.json \
        --plot-out /output/trajectory_preview.png \
        --dataset "${bag_label}" --map "${map_label}"
    ;;

  help|-h|--help)
    usage
    ;;

  *)
    usage >&2
    fail "Unknown action: ${action}"
    ;;
esac
