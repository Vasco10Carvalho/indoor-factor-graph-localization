#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${INDOOR_IMAGE_NAME:-indoor-factor-graph-localization:humble}"
DEFAULT_PROFILE="${ROOT_DIR}/configs/hospital_pose3.yaml"

usage() {
  cat <<'EOF'
Indoor Pose3/6D factor graph (ROS 2 Humble + GTSAM)

Usage:
  ./pose3_pipeline.sh build
  ./pose3_pipeline.sh check
  ./pose3_pipeline.sh run --map MAP.ply --output DIR \
    --initial-pose X Y Z ROLL PITCH YAW \
    --extrinsic X Y Z ROLL PITCH YAW [options]
  ./pose3_pipeline.sh play --bag BAG_DIRECTORY [--duration SEC] [options]
  ./pose3_pipeline.sh rviz [--domain-id ID] [--live]
  ./pose3_pipeline.sh publish --map MAP.ply --output DIR [options]
  ./pose3_pipeline.sh monitor [--domain-id ID]
  ./pose3_pipeline.sh evaluate --output DIR

Run options:
  --config YAML              Algorithm profile (default: configs/hospital_pose3.yaml)
  --domain-id ID             ROS domain (default: 76)
  --live                     Use wall time instead of rosbag /clock
  --name NAME                Docker container name
  -- ARGS                    Extra node arguments overriding the profile

Play options:
  --rate VALUE               Playback rate (default: 1.0)
  --start-offset SEC         Skip bag start (default: 0.0)
  --duration SEC             Stop cleanly after this wall-clock duration
  --topics CSV               Topic allow-list
  --domain-id ID             Must match the node and RViz

Publish saved-result options:
  --map MAP.ply              Original fixed map (mounted read-only)
  --output DIR               Directory containing indoor_pose3_6d.csv
  --map-z-max METERS         RViz-only ceiling cut (default: 2.15)
  --domain-id ID             Must match RViz (default: 76)
EOF
}

fail() { echo "[ERROR] $*" >&2; exit 2; }
need_value() { [[ $# -ge 2 ]] || fail "Missing value after $1"; }

ros_environment() {
  local domain_id="$1"
  printf '%s\0' \
    --env "ROS_DOMAIN_ID=${domain_id}" \
    --env ROS_LOCALHOST_ONLY=0 \
    --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    --env FASTDDS_BUILTIN_TRANSPORTS=UDPv4
}

action="${1:-help}"
[[ $# -eq 0 ]] || shift

case "${action}" in
  build)
    exec docker build --tag "${IMAGE_NAME}" "${ROOT_DIR}"
    ;;

  check)
    exec docker run --rm "${IMAGE_NAME}" bash -lc \
      "python3 -m py_compile /workspace/indoor_localization_pose3/*.py && \
       python3 -m unittest discover -s /workspace/tests -p 'test_*.py'"
    ;;

  run)
    map_ply=""; output=""; profile="${DEFAULT_PROFILE}"; domain_id=76
    use_sim_time=true; container_name=indoor-pose3-6d
    initial=(); extrinsic=(); extra_args=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --map) need_value "$@"; map_ply="$2"; shift 2 ;;
        --output) need_value "$@"; output="$2"; shift 2 ;;
        --config) need_value "$@"; profile="$2"; shift 2 ;;
        --domain-id) need_value "$@"; domain_id="$2"; shift 2 ;;
        --name) need_value "$@"; container_name="$2"; shift 2 ;;
        --initial-pose)
          [[ $# -ge 7 ]] || fail "--initial-pose requires X Y Z ROLL PITCH YAW"
          initial=("$2" "$3" "$4" "$5" "$6" "$7"); shift 7 ;;
        --extrinsic)
          [[ $# -ge 7 ]] || fail "--extrinsic requires X Y Z ROLL PITCH YAW"
          extrinsic=("$2" "$3" "$4" "$5" "$6" "$7"); shift 7 ;;
        --live) use_sim_time=false; shift ;;
        --) shift; extra_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown run option: $1" ;;
      esac
    done
    [[ -f "${map_ply}" ]] || fail "PLY not found: ${map_ply:-<missing>}"
    [[ -f "${profile}" ]] || fail "Profile not found: ${profile}"
    [[ -n "${output}" ]] || fail "run requires --output"
    [[ ${#initial[@]} -eq 6 ]] || fail "run requires --initial-pose with 6 values"
    [[ ${#extrinsic[@]} -eq 6 ]] || fail "run requires --extrinsic with 6 values"
    map_ply="$(realpath "${map_ply}")"; profile="$(realpath "${profile}")"
    mkdir -p "${output}"; output="$(realpath "${output}")"
    mapfile -d '' ros_env < <(ros_environment "${domain_id}")
    exec docker run --rm --init --name "${container_name}" \
      --network host --ipc host --user "$(id -u):$(id -g)" \
      --env HOME=/tmp/indoor_pose3_home "${ros_env[@]}" \
      --volume "${map_ply}:/data/map.ply:ro" \
      --volume "${profile}:/data/profile.yaml:ro" \
      --volume "${output}:/output" \
      "${IMAGE_NAME}" \
      python3 /workspace/indoor_localization_pose3/online_indoor_factor_graph_pose3.py \
        --config /data/profile.yaml --map-ply /data/map.ply --output-dir /output \
        --initial-x "${initial[0]}" --initial-y "${initial[1]}" \
        --initial-z "${initial[2]}" --initial-roll-deg "${initial[3]}" \
        --initial-pitch-deg "${initial[4]}" --initial-yaw-deg "${initial[5]}" \
        --base-to-lidar-x "${extrinsic[0]}" --base-to-lidar-y "${extrinsic[1]}" \
        --base-to-lidar-z "${extrinsic[2]}" \
        --base-to-lidar-roll-deg "${extrinsic[3]}" \
        --base-to-lidar-pitch-deg "${extrinsic[4]}" \
        --base-to-lidar-yaw-deg "${extrinsic[5]}" \
        "${extra_args[@]}" --ros-args -p use_sim_time:="${use_sim_time}"
    ;;

  play)
    bag=""; domain_id=76; rate=1.0; start_offset=0.0; duration=""
    topics="/ouster/points,/odometry/filtered,/tf,/tf_static"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --bag) need_value "$@"; bag="$2"; shift 2 ;;
        --domain-id) need_value "$@"; domain_id="$2"; shift 2 ;;
        --rate) need_value "$@"; rate="$2"; shift 2 ;;
        --start-offset) need_value "$@"; start_offset="$2"; shift 2 ;;
        --duration) need_value "$@"; duration="$2"; shift 2 ;;
        --topics) need_value "$@"; topics="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown play option: $1" ;;
      esac
    done
    [[ -f "${bag}/metadata.yaml" ]] || fail "metadata.yaml not found in ${bag:-<missing>}"
    bag="$(realpath "${bag}")"
    IFS=',' read -r -a topic_list <<< "${topics}"
    timeout_args=()
    [[ -z "${duration}" ]] || timeout_args=(timeout --signal=INT --kill-after=5 "${duration}s")
    mapfile -d '' ros_env < <(ros_environment "${domain_id}")
    exec docker run --rm --init --name indoor-pose3-player \
      --network host --ipc host "${ros_env[@]}" \
      --volume "${bag}:/data/bag:ro" "${IMAGE_NAME}" \
      "${timeout_args[@]}" ros2 bag play /data/bag --clock 100 \
        --rate "${rate}" --start-offset "${start_offset}" --delay 2 \
        --read-ahead-queue-size 10000 \
        --topics "${topic_list[@]}"
    ;;

  rviz)
    domain_id=76; use_sim_time=true
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --domain-id) need_value "$@"; domain_id="$2"; shift 2 ;;
        --live) use_sim_time=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown rviz option: $1" ;;
      esac
    done
    [[ -n "${DISPLAY:-}" ]] || fail "DISPLAY is not set"
    mapfile -d '' ros_env < <(ros_environment "${domain_id}")
    gui=(--env "DISPLAY=${DISPLAY}" --env QT_X11_NO_MITSHM=1 --volume /tmp/.X11-unix:/tmp/.X11-unix:rw)
    xauthority="${XAUTHORITY:-${HOME}/.Xauthority}"
    [[ ! -f "${xauthority}" ]] || gui+=(--env XAUTHORITY=/tmp/.docker.xauth --volume "${xauthority}:/tmp/.docker.xauth:ro")
    device=(); [[ ! -e /dev/dri ]] || device=(--device /dev/dri)
    exec docker run --rm --init --network host --ipc host \
      --user "$(id -u):$(id -g)" --env HOME=/tmp/pose3_rviz_home \
      "${ros_env[@]}" "${gui[@]}" "${device[@]}" "${IMAGE_NAME}" rviz2 \
      -d /workspace/indoor_localization_pose3/indoor_pose3.rviz \
      --ros-args -p use_sim_time:="${use_sim_time}"
    ;;

  publish)
    map_ply=""; output=""; domain_id=76; map_z_max=2.15
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --map) need_value "$@"; map_ply="$2"; shift 2 ;;
        --output) need_value "$@"; output="$2"; shift 2 ;;
        --map-z-max) need_value "$@"; map_z_max="$2"; shift 2 ;;
        --domain-id) need_value "$@"; domain_id="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown publish option: $1" ;;
      esac
    done
    [[ -f "${map_ply}" ]] || fail "PLY not found: ${map_ply:-<missing>}"
    [[ -f "${output}/indoor_pose3_6d.csv" ]] || \
      fail "Pose3 CSV not found in ${output:-<missing>}"
    map_ply="$(realpath "${map_ply}")"; output="$(realpath "${output}")"
    mapfile -d '' ros_env < <(ros_environment "${domain_id}")
    exec docker run --rm --init --network host --ipc host \
      --user "$(id -u):$(id -g)" --env HOME=/tmp/pose3_publish_home \
      "${ros_env[@]}" \
      --volume "${map_ply}:/data/map.ply:ro" \
      --volume "${output}:/output:ro" \
      "${IMAGE_NAME}" \
      python3 /workspace/indoor_localization_pose3/publish_pose3_result.py \
        --map-ply /data/map.ply --csv /output/indoor_pose3_6d.csv \
        --map-publish-z-max "${map_z_max}"
    ;;

  monitor)
    domain_id=76
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --domain-id) need_value "$@"; domain_id="$2"; shift 2 ;;
        *) fail "Unknown monitor option: $1" ;;
      esac
    done
    mapfile -d '' ros_env < <(ros_environment "${domain_id}")
    exec docker run --rm --init --network host "${ros_env[@]}" "${IMAGE_NAME}" \
      python3 /workspace/indoor_localization_pose3/monitor_status.py
    ;;

  evaluate)
    output=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --output) need_value "$@"; output="$2"; shift 2 ;;
        *) fail "Unknown evaluate option: $1" ;;
      esac
    done
    [[ -f "${output}/indoor_pose3_6d.csv" ]] || fail "Pose3 CSV not found in ${output:-<missing>}"
    output="$(realpath "${output}")"
    exec docker run --rm --user "$(id -u):$(id -g)" \
      --env HOME=/tmp/pose3_eval_home --volume "${output}:/output" "${IMAGE_NAME}" \
      python3 /workspace/indoor_localization_pose3/evaluate_pose3.py \
        --csv /output/indoor_pose3_6d.csv \
        --output-json /output/indoor_pose3_6d_evaluation.json \
        --output-plot /output/indoor_pose3_6d_evaluation.png
    ;;

  -h|--help|help) usage ;;
  *) fail "Unknown action: ${action}" ;;
esac
