#!/usr/bin/env python3
"""Robust incremental indoor Pose2 localization on a fixed occupancy map."""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time

import gtsam
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMsg
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import ColorRGBA, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker
import yaml

from indoor_loop_closer import LoopCloser, LoopMatch
from indoor_scan_matcher import (
    MatchResult,
    OccupancyMap,
    global_scan_match,
    laser_scan_points,
    local_scan_match,
    wrap_angle,
)
from profile_config import apply_profile_defaults


def stamp_ns(message) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def set_stamp(stamp, value_ns: int) -> None:
    stamp.sec = int(value_ns // 1_000_000_000)
    stamp.nanosec = int(value_ns % 1_000_000_000)


def quaternion_to_yaw(quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2),
    )


def nearest_sample(samples, target_ns: int, maximum_difference_sec: float):
    if not samples:
        return None
    sample = min(samples, key=lambda value: abs(value.stamp_ns - target_ns))
    if abs(sample.stamp_ns - target_ns) / 1e9 > maximum_difference_sec:
        return None
    return sample


def integrate_gyro_yaw(
    samples: tuple["GyroSample", ...], start_ns: int, end_ns: int
) -> float | None:
    if end_ns <= start_ns or len(samples) < 2:
        return None
    times = np.asarray([sample.stamp_ns for sample in samples], dtype=np.int64)
    rates = np.asarray([sample.angular_velocity_z for sample in samples], dtype=float)
    if times[0] > start_ns or times[-1] < end_ns:
        return None
    inside = (times > start_ns) & (times < end_ns)
    selected_times = np.concatenate(([start_ns], times[inside], [end_ns]))
    selected_rates = np.interp(selected_times.astype(float), times.astype(float), rates)
    return float(np.trapz(selected_rates, selected_times.astype(float) / 1e9))


def robust_noise(sigmas: np.ndarray, huber_k: float):
    diagonal = gtsam.noiseModel.Diagonal.Sigmas(sigmas.astype(float))
    estimator = gtsam.noiseModel.mEstimator.Huber.Create(float(huber_k))
    return gtsam.noiseModel.Robust.Create(estimator, diagonal)


def dcs_noise(sigmas: np.ndarray, phi: float):
    diagonal = gtsam.noiseModel.Diagonal.Sigmas(sigmas.astype(float))
    estimator = gtsam.noiseModel.mEstimator.DCS.Create(float(phi))
    return gtsam.noiseModel.Robust.Create(estimator, diagonal)


def align_odometry_pose(
    map_start: gtsam.Pose2,
    odom_start: gtsam.Pose2,
    odom_current: gtsam.Pose2,
) -> gtsam.Pose2:
    """Express odometry in map coordinates without applying map corrections."""
    return map_start.compose(odom_start.between(odom_current))


def rate_hz(samples: deque[float]) -> float:
    if len(samples) < 2:
        return 0.0
    elapsed = samples[-1] - samples[0]
    return (len(samples) - 1) / elapsed if elapsed > 0.0 else 0.0


@dataclass(frozen=True)
class OdomSample:
    stamp_ns: int
    pose: gtsam.Pose2


@dataclass(frozen=True)
class ImuSample:
    stamp_ns: int
    yaw: float


@dataclass(frozen=True)
class GyroSample:
    stamp_ns: int
    angular_velocity_z: float


@dataclass(frozen=True)
class AmclSample:
    stamp_ns: int
    x: float
    y: float
    yaw: float
    sigma_x: float
    sigma_y: float
    sigma_yaw: float


class IndoorPose2FactorGraph(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("indoor_factor_graph_pose2_dcs")
        self.args = args
        self.map = OccupancyMap(args.map_yaml)
        self.keyframe_period_ns = int(1e9 / args.keyframe_rate_hz)

        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(args.relinearize_threshold)
        parameters.relinearizeSkip = args.relinearize_skip
        self.isam = gtsam.ISAM2(parameters)
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    args.prior_sigma_xy,
                    args.prior_sigma_xy,
                    math.radians(args.prior_sigma_yaw_deg),
                ]
            )
        )
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    args.odom_sigma_xy,
                    args.odom_sigma_xy,
                    math.radians(args.odom_sigma_yaw_deg),
                ]
            )
        )
        self.weak_odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    args.weak_odom_sigma_xy,
                    args.weak_odom_sigma_xy,
                    math.radians(args.weak_odom_sigma_yaw_deg),
                ]
            )
        )
        self.robot_imu_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    args.imu_translation_sigma,
                    args.imu_translation_sigma,
                    math.radians(args.robot_imu_delta_yaw_sigma_deg),
                ]
            )
        )
        self.lidar_gyro_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    args.imu_translation_sigma,
                    args.imu_translation_sigma,
                    math.radians(args.lidar_gyro_delta_yaw_sigma_deg),
                ]
            )
        )
        self.loop_closer = LoopCloser(
            enabled=args.enable_loop_closure,
            check_interval=args.loop_check_interval,
            minimum_separation=args.loop_minimum_separation,
            candidate_radius=args.loop_candidate_radius,
            maximum_candidates=args.loop_maximum_candidates,
            submap_half_width=args.loop_submap_half_width,
            voxel_size=args.loop_voxel_size,
            maximum_correspondence_distance=args.loop_max_correspondence_distance,
            minimum_inlier_ratio=args.loop_minimum_inlier_ratio,
            maximum_rmse=args.loop_maximum_rmse,
            minimum_correspondences=args.loop_minimum_correspondences,
            maximum_correction=args.loop_maximum_correction,
            maximum_correction_yaw_deg=args.loop_maximum_correction_yaw_deg,
            confirmation_count=args.loop_confirmation_count,
            confirmation_index_tolerance=args.loop_confirmation_index_tolerance,
        )

        self.buffer_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.odom_buffer: deque[OdomSample] = deque(maxlen=args.buffer_size)
        self.imu_buffer: deque[ImuSample] = deque(maxlen=args.buffer_size)
        self.lidar_gyro_buffer: deque[GyroSample] = deque(maxlen=args.buffer_size)
        self.amcl_history: deque[AmclSample] = deque(
            maxlen=max(20, args.amcl_consistency_samples * 4)
        )
        self.pending_scan: LaserScan | None = None
        self.pending_event = threading.Event()
        self.stop_event = threading.Event()

        self.initial_pose: tuple[float, float, float] | None = None
        if args.initial_x is not None:
            self.initial_pose = (
                args.initial_x,
                args.initial_y,
                math.radians(args.initial_yaw_deg),
            )
        self.latest_pose: gtsam.Pose2 | None = None
        self.last_keyframe_odom: OdomSample | None = None
        self.last_keyframe_imu: ImuSample | None = None
        self.last_keyframe_scan_ns: int | None = None
        self.keyframe_index = -1
        self.last_processed_scan_ns: int | None = None
        self.path_poses: list[tuple[int, gtsam.Pose2]] = []
        self.odom_only_path_poses: list[tuple[int, gtsam.Pose2]] = []
        self.odom_only_map_start: gtsam.Pose2 | None = None
        self.odom_only_odom_start: gtsam.Pose2 | None = None
        self.last_odom_only_stamp_ns: int | None = None
        self.loop_edges: list[tuple[int, int]] = []
        self.recovery_indices: set[int] = set()
        self.last_match: MatchResult | None = None
        self.last_loop_match: LoopMatch | None = None
        self.last_scan_correction_xy = 0.0
        self.last_scan_correction_yaw_deg = 0.0
        self.last_rejection_reasons: list[str] = []
        self.rejection_reasons: dict[str, int] = {}
        self.last_status = "waiting_for_inputs"
        self.consecutive_scan_rejections = 0
        self.recovery_hypothesis: tuple[gtsam.Pose2, OdomSample, int] | None = None
        self.recovery_attempts = 0
        self.recoveries = 0
        self.last_amcl_recovery_index = -args.amcl_recovery_cooldown_keyframes
        self.last_amcl_attempt_index = -args.amcl_recovery_retry_keyframes
        self.amcl_messages_received = 0
        self.amcl_recovery_attempts = 0
        self.amcl_recovery_rejections = 0
        self.amcl_recoveries = 0
        self.amcl_factors = 0
        self.last_amcl_mahalanobis_sq: float | None = None
        self.last_amcl_sigma: tuple[float, float, float] | None = None

        self.scans_received = 0
        self.scans_processed = 0
        self.scans_accepted = 0
        self.scans_rejected = 0
        self.scans_dropped = 0
        self.odom_jumps_rejected = 0
        self.loop_checks = 0
        self.loop_closures = 0
        self.robot_imu_factors = 0
        self.lidar_gyro_factors = 0
        self.scan_rx_wall: deque[float] = deque(maxlen=100)
        self.odom_rx_wall: deque[float] = deque(maxlen=200)
        self.imu_rx_wall: deque[float] = deque(maxlen=300)
        self.lidar_imu_rx_wall: deque[float] = deque(maxlen=300)
        self.scan_rx_stamp: deque[float] = deque(maxlen=100)
        self.odom_rx_stamp: deque[float] = deque(maxlen=200)
        self.imu_rx_stamp: deque[float] = deque(maxlen=300)
        self.lidar_imu_rx_stamp: deque[float] = deque(maxlen=300)
        self.processing_wall: deque[float] = deque(maxlen=100)
        self.processing_ms: deque[float] = deque(maxlen=200)

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = args.output_csv.open("w", encoding="utf-8", newline="")
        self.output_writer = csv.writer(self.output_file)
        self.output_writer.writerow(
            [
                "stamp_ns",
                "map_x",
                "map_y",
                "map_z",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "scan_status",
                "scan_inlier_ratio",
                "scan_mean_distance",
                "scan_valid_ratio",
                "scan_correction_xy",
                "scan_correction_yaw_deg",
                "rejection_reasons",
                "loop_candidate_index",
                "loop_rmse",
                "loop_inlier_ratio",
            ]
        )
        args.output_odom_only_csv.parent.mkdir(parents=True, exist_ok=True)
        self.odom_only_output_file = args.output_odom_only_csv.open(
            "w", encoding="utf-8", newline=""
        )
        self.odom_only_output_writer = csv.writer(self.odom_only_output_file)
        self.odom_only_output_writer.writerow(
            [
                "stamp_ns",
                "map_x",
                "map_y",
                "yaw_deg",
                "source_odom_x",
                "source_odom_y",
                "source_odom_yaw_deg",
            ]
        )
        args.output_status_json.parent.mkdir(parents=True, exist_ok=True)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Odometry, args.odom_topic, self.odom_callback, sensor_qos)
        self.create_subscription(Imu, args.imu_topic, self.imu_callback, sensor_qos)
        self.create_subscription(
            Imu, args.lidar_imu_topic, self.lidar_imu_callback, sensor_qos
        )
        self.create_subscription(LaserScan, args.scan_topic, self.scan_callback, sensor_qos)
        self.create_subscription(
            PoseWithCovarianceStamped,
            args.amcl_pose_topic,
            self.amcl_callback,
            output_qos,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            args.initialpose_topic,
            self.initialpose_callback,
            output_qos,
        )
        self.odom_publisher = self.create_publisher(
            Odometry, args.output_odom_topic, output_qos
        )
        self.path_publisher = self.create_publisher(
            PathMsg, args.output_path_topic, latched_qos
        )
        self.segments_publisher = self.create_publisher(
            Marker, args.output_segments_topic, latched_qos
        )
        self.recovery_markers_publisher = self.create_publisher(
            Marker, args.output_recovery_markers_topic, latched_qos
        )
        self.odom_only_path_publisher = self.create_publisher(
            PathMsg, args.output_odom_only_path_topic, latched_qos
        )
        self.axes_publisher = self.create_publisher(
            Marker, args.output_axes_topic, output_qos
        )
        self.loops_publisher = self.create_publisher(
            Marker, args.output_loops_topic, latched_qos
        )
        self.status_publisher = self.create_publisher(
            String, args.output_status_topic, output_qos
        )
        self.map_publisher = self.create_publisher(
            OccupancyGrid, args.map_topic, latched_qos
        )
        self.amcl_seed_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            args.amcl_initialpose_topic,
            output_qos,
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.map_published = False
        # Sim-time timers do not fire before the bag starts publishing /clock.
        # Publish immediately so RViz can set /initialpose before playback.
        self.publish_map_once()
        self.create_timer(1.0 / args.path_publish_rate_hz, self.publish_path)
        self.create_timer(1.0 / args.status_publish_rate_hz, self.publish_status)
        self.create_timer(1.0, self.publish_map_once)

        self.worker = threading.Thread(target=self.scan_worker, daemon=True)
        self.worker.start()
        if self.initial_pose:
            mode = "configured manual pose"
        elif args.wait_for_initial_pose:
            mode = "waiting for RViz /initialpose"
        else:
            mode = "global scan search"
        self.get_logger().info(
            "Indoor Pose2 DCS graph ready: "
            f"map={args.map_yaml}, odom={args.odom_topic}, imu={args.imu_topic}, "
            f"lidar_imu={args.lidar_imu_topic}, scan={args.scan_topic}, "
            f"initialization={mode}"
        )
        self.get_logger().info(
            "AMCL covariance recovery="
            f"{'on' if args.enable_amcl_recovery else 'off'}; "
            "AMCL TF output must remain disabled"
        )

    def odom_callback(self, message: Odometry) -> None:
        self.odom_rx_wall.append(time.monotonic())
        self.odom_rx_stamp.append(stamp_ns(message) / 1e9)
        pose = message.pose.pose
        sample = OdomSample(
            stamp_ns(message),
            gtsam.Pose2(
                float(pose.position.x),
                float(pose.position.y),
                quaternion_to_yaw(pose.orientation),
            ),
        )
        with self.buffer_lock:
            self.odom_buffer.append(sample)
        self.record_odom_only_pose(sample)
        self.publish_propagated_odometry(sample, message)

    def record_odom_only_pose(self, odom: OdomSample) -> None:
        with self.state_lock:
            if self.initial_pose is None:
                return
            if self.odom_only_odom_start is None:
                self.odom_only_odom_start = odom.pose
                self.odom_only_map_start = gtsam.Pose2(*self.initial_pose)
            if (
                self.last_odom_only_stamp_ns is not None
                and odom.stamp_ns - self.last_odom_only_stamp_ns
                < self.keyframe_period_ns
            ):
                return
            aligned = align_odometry_pose(
                self.odom_only_map_start,
                self.odom_only_odom_start,
                odom.pose,
            )
            self.odom_only_path_poses.append((odom.stamp_ns, aligned))
            self.last_odom_only_stamp_ns = odom.stamp_ns
        self.odom_only_output_writer.writerow(
            [
                odom.stamp_ns,
                aligned.x(),
                aligned.y(),
                math.degrees(aligned.theta()),
                odom.pose.x(),
                odom.pose.y(),
                math.degrees(odom.pose.theta()),
            ]
        )
        self.odom_only_output_file.flush()

    def imu_callback(self, message: Imu) -> None:
        self.imu_rx_wall.append(time.monotonic())
        self.imu_rx_stamp.append(stamp_ns(message) / 1e9)
        sample = ImuSample(stamp_ns(message), quaternion_to_yaw(message.orientation))
        with self.buffer_lock:
            self.imu_buffer.append(sample)

    def lidar_imu_callback(self, message: Imu) -> None:
        self.lidar_imu_rx_wall.append(time.monotonic())
        self.lidar_imu_rx_stamp.append(stamp_ns(message) / 1e9)
        sample = GyroSample(stamp_ns(message), float(message.angular_velocity.z))
        with self.buffer_lock:
            self.lidar_gyro_buffer.append(sample)

    def amcl_callback(self, message: PoseWithCovarianceStamped) -> None:
        pose = message.pose.pose
        covariance = message.pose.covariance
        sample = AmclSample(
            stamp_ns=stamp_ns(message),
            x=float(pose.position.x),
            y=float(pose.position.y),
            yaw=quaternion_to_yaw(pose.orientation),
            sigma_x=math.sqrt(max(0.0, float(covariance[0]))),
            sigma_y=math.sqrt(max(0.0, float(covariance[7]))),
            sigma_yaw=math.sqrt(max(0.0, float(covariance[35]))),
        )
        if not all(
            math.isfinite(value)
            for value in (
                sample.x,
                sample.y,
                sample.yaw,
                sample.sigma_x,
                sample.sigma_y,
                sample.sigma_yaw,
            )
        ):
            return
        with self.state_lock:
            self.amcl_history.append(sample)
        self.amcl_messages_received += 1

    def initialpose_callback(self, message: PoseWithCovarianceStamped) -> None:
        pose = message.pose.pose
        with self.state_lock:
            if self.keyframe_index >= 0:
                self.get_logger().warning(
                    "Ignoring /initialpose after initialization; restart to relocalize"
                )
                return
            self.initial_pose = (
                float(pose.position.x),
                float(pose.position.y),
                quaternion_to_yaw(pose.orientation),
            )
        self.get_logger().info(
            f"Received initial pose x={pose.position.x:.2f}, y={pose.position.y:.2f}, "
            f"yaw={math.degrees(self.initial_pose[2]):.1f} deg"
        )

    def scan_callback(self, message: LaserScan) -> None:
        self.scan_rx_wall.append(time.monotonic())
        self.scan_rx_stamp.append(stamp_ns(message) / 1e9)
        self.scans_received += 1
        with self.pending_lock:
            if self.pending_scan is not None:
                self.scans_dropped += 1
            self.pending_scan = message
            self.pending_event.set()

    def scan_worker(self) -> None:
        while not self.stop_event.is_set():
            if not self.pending_event.wait(0.2):
                continue
            with self.pending_lock:
                message = self.pending_scan
                self.pending_scan = None
                self.pending_event.clear()
            if message is None:
                continue
            try:
                self.process_scan(message)
            except Exception as exc:
                self.last_status = "processing_exception"
                self.get_logger().error(f"Indoor Pose2 processing failed: {exc}")

    def process_scan(self, message: LaserScan) -> None:
        started = time.monotonic()
        scan_stamp = stamp_ns(message)
        if (
            self.last_processed_scan_ns is not None
            and scan_stamp - self.last_processed_scan_ns < self.keyframe_period_ns
        ):
            return
        with self.buffer_lock:
            odom = nearest_sample(
                tuple(self.odom_buffer), scan_stamp, self.args.odom_max_sync_sec
            )
            imu = nearest_sample(
                tuple(self.imu_buffer), scan_stamp, self.args.imu_max_sync_sec
            )
            lidar_gyro_samples = tuple(self.lidar_gyro_buffer)
        if odom is None or (imu is None and not self.args.allow_missing_robot_imu):
            self.last_status = (
                "waiting_for_synchronized_odom"
                if odom is None and self.args.allow_missing_robot_imu
                else "waiting_for_synchronized_odom_imu"
            )
            return

        points = laser_scan_points(
            np.asarray(message.ranges),
            message.angle_min,
            message.angle_increment,
            message.range_min,
            message.range_max,
            self.args.maximum_scan_range,
            self.args.maximum_scan_points,
        )
        with self.state_lock:
            latest = self.latest_pose
            previous_odom = self.last_keyframe_odom
            previous_imu = self.last_keyframe_imu
            previous_scan_ns = self.last_keyframe_scan_ns
            initial = self.initial_pose

        if initial is None and self.args.wait_for_initial_pose:
            self.last_status = "waiting_for_initial_pose"
            return

        robot_imu_delta = None
        lidar_gyro_delta = None
        if imu is not None and previous_imu is not None:
            robot_imu_delta = float(wrap_angle(imu.yaw - previous_imu.yaw))
        if self.args.use_lidar_gyro_factor and previous_scan_ns is not None:
            lidar_gyro_delta = integrate_gyro_yaw(
                lidar_gyro_samples, previous_scan_ns, scan_stamp
            )
            if lidar_gyro_delta is not None:
                interval = (scan_stamp - previous_scan_ns) / 1e9
                lidar_gyro_delta = float(
                    wrap_angle(
                        lidar_gyro_delta
                        - self.args.lidar_gyro_bias_z_rad_s * interval
                    )
                )
        maximum_imu_delta = math.radians(self.args.maximum_imu_delta_yaw_deg)
        if robot_imu_delta is not None and (
            not math.isfinite(robot_imu_delta)
            or abs(robot_imu_delta) > maximum_imu_delta
        ):
            robot_imu_delta = None
        if lidar_gyro_delta is not None and (
            not math.isfinite(lidar_gyro_delta)
            or abs(lidar_gyro_delta) > maximum_imu_delta
        ):
            lidar_gyro_delta = None

        odom_delta = None
        odom_delta_valid = True
        if latest is None or previous_odom is None:
            if initial is None:
                match = global_scan_match(
                    self.map,
                    points,
                    particle_count=self.args.global_particles,
                    seed=self.args.global_seed,
                    truncation_distance=self.args.truncation_distance,
                )
                initial = (match.x, match.y, match.yaw)
                with self.state_lock:
                    self.initial_pose = initial
            predicted = gtsam.Pose2(*initial)
            if "match" not in locals():
                match = local_scan_match(
                    self.map,
                    points,
                    np.asarray(initial),
                    coarse_xy_window=self.args.local_xy_window,
                    coarse_xy_step=self.args.local_xy_step,
                    coarse_yaw_window_deg=self.args.local_yaw_window_deg,
                    coarse_yaw_step_deg=self.args.local_yaw_step_deg,
                    truncation_distance=self.args.truncation_distance,
                )
        else:
            odom_delta = previous_odom.pose.between(odom.pose)
            odom_delta_valid = self.is_reasonable_odom_delta(odom_delta)
            predicted = latest.compose(odom_delta) if odom_delta_valid else latest
            if not odom_delta_valid:
                self.odom_jumps_rejected += 1
                self.get_logger().warning(
                    "Rejected raw wheel-odometry discontinuity; scan-to-map will recover"
                )
            match = local_scan_match(
                self.map,
                points,
                np.asarray([predicted.x(), predicted.y(), predicted.theta()]),
                coarse_xy_window=self.args.local_xy_window,
                coarse_xy_step=self.args.local_xy_step,
                coarse_yaw_window_deg=self.args.local_yaw_window_deg,
                coarse_yaw_step_deg=self.args.local_yaw_step_deg,
                truncation_distance=self.args.truncation_distance,
            )

        measured = gtsam.Pose2(match.x, match.y, match.yaw)
        correction = predicted.between(measured)
        correction_xy = math.hypot(correction.x(), correction.y())
        correction_yaw_deg = abs(math.degrees(correction.theta()))
        rejection_reasons = []
        if match.inlier_ratio < self.args.minimum_inlier_ratio:
            rejection_reasons.append("inlier_ratio")
        if match.mean_distance > self.args.maximum_mean_distance:
            rejection_reasons.append("mean_distance")
        if match.valid_ratio < self.args.minimum_valid_ratio:
            rejection_reasons.append("valid_ratio")
        if correction_xy > self.args.maximum_scan_correction:
            rejection_reasons.append("correction_xy")
        if correction_yaw_deg > self.args.maximum_scan_rotation_deg:
            rejection_reasons.append("correction_yaw")
        accepted = not rejection_reasons
        recovered = False
        recovery_source: str | None = None
        amcl_factor: AmclSample | None = None
        if accepted:
            self.consecutive_scan_rejections = 0
            self.recovery_hypothesis = None
        else:
            self.consecutive_scan_rejections += 1
            amcl_recovery = self.try_amcl_recovery(scan_stamp, points, predicted)
            if amcl_recovery is not None:
                recovery_match, amcl_factor = amcl_recovery
                recovery_source = "amcl"
            else:
                recovery_match = self.try_scan_recovery(points, predicted, odom)
                if recovery_match is not None:
                    recovery_source = "scan"
            if recovery_match is not None:
                match = recovery_match
                measured = gtsam.Pose2(match.x, match.y, match.yaw)
                correction = predicted.between(measured)
                correction_xy = math.hypot(correction.x(), correction.y())
                correction_yaw_deg = abs(math.degrees(correction.theta()))
                rejection_reasons = []
                accepted = True
                recovered = True
                self.consecutive_scan_rejections = 0
                self.recovery_hypothesis = None
                if recovery_source == "amcl":
                    self.amcl_recoveries += 1
                    self.last_amcl_recovery_index = self.keyframe_index
                else:
                    self.recoveries += 1
        optimized, loop_match = self.update_graph(
            odom,
            imu,
            predicted,
            measured if accepted else None,
            match,
            odom_delta,
            odom_delta_valid,
            robot_imu_delta,
            lidar_gyro_delta,
            points,
            scan_stamp,
            recovered,
            amcl_factor,
        )
        self.publish_map_to_odom_transform(scan_stamp, optimized, odom.pose)
        if self.keyframe_index == 0 and self.args.enable_amcl_recovery:
            self.publish_amcl_seed(scan_stamp, optimized)
        self.last_processed_scan_ns = scan_stamp
        self.last_match = match
        self.last_loop_match = loop_match
        self.last_scan_correction_xy = correction_xy
        self.last_scan_correction_yaw_deg = correction_yaw_deg
        self.last_rejection_reasons = rejection_reasons
        for reason in rejection_reasons:
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        self.last_status = (
            "recovered_amcl"
            if recovery_source == "amcl"
            else "accepted_recovery"
            if recovery_source == "scan"
            else "accepted_loop"
            if loop_match is not None
            else "accepted" if accepted else "rejected_scan_match"
        )
        self.scans_processed += 1
        self.scans_accepted += int(accepted)
        self.scans_rejected += int(not accepted)
        self.processing_wall.append(time.monotonic())
        self.processing_ms.append((time.monotonic() - started) * 1000.0)
        self.write_output(scan_stamp, optimized, self.last_status, match, loop_match)
        if self.scans_processed % 20 == 0:
            self.get_logger().info(
                f"k={self.keyframe_index} {self.last_status} "
                f"xyyaw=({optimized.x():.2f},{optimized.y():.2f},"
                f"{math.degrees(optimized.theta()):.1f}) "
                f"inliers={match.inlier_ratio:.2f} mean={match.mean_distance:.2f}m "
                f"total={self.processing_ms[-1]:.1f}ms"
            )

    def try_amcl_recovery(
        self,
        scan_stamp: int,
        points: np.ndarray,
        predicted: gtsam.Pose2,
    ) -> tuple[MatchResult, AmclSample] | None:
        if (
            not self.args.enable_amcl_recovery
            or self.consecutive_scan_rejections
            < self.args.amcl_recovery_rejections
            or self.keyframe_index - self.last_amcl_recovery_index
            < self.args.amcl_recovery_cooldown_keyframes
            or self.keyframe_index - self.last_amcl_attempt_index
            < self.args.amcl_recovery_retry_keyframes
        ):
            return None

        with self.state_lock:
            history = tuple(self.amcl_history)
        recent = [
            sample
            for sample in history
            if abs(scan_stamp - sample.stamp_ns) / 1e9 <= self.args.amcl_max_age_sec
        ]
        if len(recent) < self.args.amcl_consistency_samples:
            return None
        recent = recent[-self.args.amcl_consistency_samples :]
        candidate = recent[-1]
        if (
            max(candidate.sigma_x, candidate.sigma_y) > self.args.amcl_max_sigma_xy
            or candidate.sigma_yaw
            > math.radians(self.args.amcl_max_sigma_yaw_deg)
        ):
            return None
        for sample in recent[:-1]:
            if (
                math.hypot(sample.x - candidate.x, sample.y - candidate.y)
                > self.args.amcl_consistency_xy
                or abs(wrap_angle(sample.yaw - candidate.yaw))
                > math.radians(self.args.amcl_consistency_yaw_deg)
            ):
                return None

        self.last_amcl_attempt_index = self.keyframe_index
        self.amcl_recovery_attempts += 1
        amcl_pose = gtsam.Pose2(candidate.x, candidate.y, candidate.yaw)
        innovation_pose = predicted.between(amcl_pose)
        innovation = np.asarray(
            [innovation_pose.x(), innovation_pose.y(), innovation_pose.theta()],
            dtype=float,
        )
        displacement = math.hypot(innovation_pose.x(), innovation_pose.y())
        yaw_difference = abs(math.degrees(innovation_pose.theta()))
        if (
            displacement > self.args.amcl_recovery_max_jump
            or yaw_difference > self.args.amcl_recovery_max_yaw_difference_deg
        ):
            self.amcl_recovery_rejections += 1
            return None

        inflation = self.args.amcl_covariance_inflation
        amcl_covariance = np.diag(
            [
                max(candidate.sigma_x, self.args.amcl_min_sigma_xy) ** 2,
                max(candidate.sigma_y, self.args.amcl_min_sigma_xy) ** 2,
                max(
                    candidate.sigma_yaw,
                    math.radians(self.args.amcl_min_sigma_yaw_deg),
                )
                ** 2,
            ]
        ) * inflation
        try:
            graph_covariance = np.asarray(
                self.isam.marginalCovariance(gtsam.symbol("x", self.keyframe_index)),
                dtype=float,
            )
        except (IndexError, RuntimeError, ValueError):
            graph_covariance = np.diag(
                [
                    self.args.weak_odom_sigma_xy**2,
                    self.args.weak_odom_sigma_xy**2,
                    math.radians(self.args.weak_odom_sigma_yaw_deg) ** 2,
                ]
            )
        innovation_covariance = graph_covariance + amcl_covariance
        mahalanobis_sq = float(
            innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation
        )
        self.last_amcl_mahalanobis_sq = mahalanobis_sq
        self.last_amcl_sigma = (
            candidate.sigma_x,
            candidate.sigma_y,
            candidate.sigma_yaw,
        )
        if mahalanobis_sq > self.args.amcl_max_mahalanobis_sq:
            self.amcl_recovery_rejections += 1
            self.get_logger().warning(
                "AMCL recovery rejected by covariance gate: "
                f"d2={mahalanobis_sq:.1f}, jump={displacement:.2f}m/"
                f"{yaw_difference:.1f}deg"
            )
            return None

        match = local_scan_match(
            self.map,
            points,
            np.asarray([candidate.x, candidate.y, candidate.yaw], dtype=float),
            coarse_xy_window=self.args.amcl_recovery_xy_window,
            coarse_xy_step=self.args.local_xy_step,
            coarse_yaw_window_deg=self.args.amcl_recovery_yaw_window_deg,
            coarse_yaw_step_deg=self.args.local_yaw_step_deg,
            truncation_distance=self.args.truncation_distance,
        )
        refined = gtsam.Pose2(match.x, match.y, match.yaw)
        refinement = amcl_pose.between(refined)
        if (
            match.inlier_ratio < self.args.amcl_recovery_minimum_inlier_ratio
            or match.mean_distance > self.args.amcl_recovery_maximum_mean_distance
            or match.valid_ratio < self.args.amcl_recovery_minimum_valid_ratio
            or math.hypot(refinement.x(), refinement.y())
            > self.args.amcl_recovery_xy_window
            or abs(math.degrees(refinement.theta()))
            > self.args.amcl_recovery_yaw_window_deg
        ):
            self.amcl_recovery_rejections += 1
            self.get_logger().warning(
                "AMCL recovery rejected by independent scan-to-map validation"
            )
            return None

        self.get_logger().warning(
            "AMCL covariance recovery accepted: "
            f"pose=({match.x:.2f},{match.y:.2f},"
            f"{math.degrees(match.yaw):.1f}deg), d2={mahalanobis_sq:.1f}, "
            f"inliers={match.inlier_ratio:.2f}"
        )
        return match, candidate

    def try_scan_recovery(
        self, points: np.ndarray, predicted: gtsam.Pose2, odom: OdomSample
    ) -> MatchResult | None:
        if not self.args.enable_scan_recovery:
            return None
        if self.consecutive_scan_rejections < self.args.recovery_after_rejections:
            return None
        if (
            self.recovery_hypothesis is None
            and self.consecutive_scan_rejections % self.args.recovery_check_interval != 0
        ):
            return None
        self.recovery_attempts += 1
        candidate = global_scan_match(
            self.map,
            points,
            particle_count=self.args.recovery_particles,
            seed=self.args.global_seed + self.recovery_attempts,
            truncation_distance=self.args.truncation_distance,
        )
        pose = gtsam.Pose2(candidate.x, candidate.y, candidate.yaw)
        correction = predicted.between(pose)
        if (
            candidate.inlier_ratio < self.args.recovery_minimum_inlier_ratio
            or candidate.mean_distance > self.args.recovery_maximum_mean_distance
            or candidate.valid_ratio < self.args.minimum_valid_ratio
            or math.hypot(correction.x(), correction.y())
            > self.args.recovery_maximum_xy_distance
            or abs(math.degrees(correction.theta()))
            > self.args.recovery_maximum_yaw_deg
        ):
            self.recovery_hypothesis = None
            return None
        confirmations = 1
        if self.recovery_hypothesis is not None:
            previous_pose, previous_odom, previous_confirmations = self.recovery_hypothesis
            expected = previous_pose.compose(previous_odom.pose.between(odom.pose))
            discrepancy = expected.between(pose)
            if (
                math.hypot(discrepancy.x(), discrepancy.y())
                <= self.args.recovery_confirmation_xy_distance
                and abs(math.degrees(discrepancy.theta()))
                <= self.args.recovery_confirmation_yaw_deg
            ):
                confirmations = previous_confirmations + 1
        self.recovery_hypothesis = (pose, odom, confirmations)
        if confirmations < self.args.recovery_confirmations:
            return None
        self.get_logger().warning(
            f"Confirmed scan-to-map recovery after {self.consecutive_scan_rejections} "
            f"rejections: ({pose.x():.2f}, {pose.y():.2f}, "
            f"{math.degrees(pose.theta()):.1f} deg), "
            f"inliers={candidate.inlier_ratio:.2f}"
        )
        return candidate

    def update_graph(
        self,
        odom: OdomSample,
        imu: ImuSample | None,
        predicted: gtsam.Pose2,
        scan_pose: gtsam.Pose2 | None,
        match: MatchResult,
        odom_delta: gtsam.Pose2 | None,
        odom_delta_valid: bool,
        robot_imu_delta: float | None,
        lidar_gyro_delta: float | None,
        scan_points: np.ndarray,
        scan_stamp: int,
        recovered: bool,
        amcl_factor: AmclSample | None,
    ) -> tuple[gtsam.Pose2, LoopMatch | None]:
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        index = self.keyframe_index + 1
        key = gtsam.symbol("x", index)
        if index == 0:
            graph.add(gtsam.PriorFactorPose2(key, predicted, self.prior_noise))
        else:
            previous_key = gtsam.symbol("x", index - 1)
            graph.add(
                gtsam.BetweenFactorPose2(
                    previous_key,
                    key,
                    odom_delta
                    if odom_delta_valid and not recovered and odom_delta is not None
                    else gtsam.Pose2(),
                    self.odom_noise
                    if odom_delta_valid and not recovered
                    else self.weak_odom_noise,
                )
            )
            if (
                not recovered
                and self.args.use_robot_imu_factor
                and robot_imu_delta is not None
            ):
                graph.add(
                    gtsam.BetweenFactorPose2(
                        previous_key,
                        key,
                        gtsam.Pose2(0.0, 0.0, robot_imu_delta),
                        self.robot_imu_noise,
                    )
                )
                self.robot_imu_factors += 1
            if not recovered and lidar_gyro_delta is not None:
                graph.add(
                    gtsam.BetweenFactorPose2(
                        previous_key,
                        key,
                        gtsam.Pose2(0.0, 0.0, lidar_gyro_delta),
                        self.lidar_gyro_noise,
                    )
                )
                self.lidar_gyro_factors += 1
        values.insert(key, scan_pose if recovered and scan_pose is not None else predicted)

        if scan_pose is not None and amcl_factor is None:
            sigma_xy = float(
                np.clip(
                    match.mean_distance / math.sqrt(max(match.inlier_ratio, 0.05)),
                    self.args.scan_min_sigma_xy,
                    self.args.scan_max_sigma_xy,
                )
            )
            sigma_yaw = math.radians(
                float(
                    np.clip(
                        2.0 + 10.0 * (1.0 - match.inlier_ratio),
                        self.args.scan_min_sigma_yaw_deg,
                        self.args.scan_max_sigma_yaw_deg,
                    )
                )
            )
            graph.add(
                gtsam.PriorFactorPose2(
                    key,
                    scan_pose,
                    robust_noise(
                        np.asarray([sigma_xy, sigma_xy, sigma_yaw]),
                        self.args.scan_huber_k,
                    ),
                )
            )

        if amcl_factor is not None:
            sigma_scale = math.sqrt(self.args.amcl_covariance_inflation)
            sigma_x = float(
                np.clip(
                    max(amcl_factor.sigma_x, self.args.amcl_min_sigma_xy)
                    * sigma_scale,
                    self.args.amcl_min_sigma_xy * sigma_scale,
                    self.args.amcl_factor_max_sigma_xy,
                )
            )
            sigma_y = float(
                np.clip(
                    max(amcl_factor.sigma_y, self.args.amcl_min_sigma_xy)
                    * sigma_scale,
                    self.args.amcl_min_sigma_xy * sigma_scale,
                    self.args.amcl_factor_max_sigma_xy,
                )
            )
            sigma_yaw = float(
                np.clip(
                    max(
                        amcl_factor.sigma_yaw,
                        math.radians(self.args.amcl_min_sigma_yaw_deg),
                    )
                    * sigma_scale,
                    math.radians(self.args.amcl_min_sigma_yaw_deg) * sigma_scale,
                    math.radians(self.args.amcl_factor_max_sigma_yaw_deg),
                )
            )
            graph.add(
                gtsam.PriorFactorPose2(
                    key,
                    gtsam.Pose2(amcl_factor.x, amcl_factor.y, amcl_factor.yaw),
                    robust_noise(
                        np.asarray([sigma_x, sigma_y, sigma_yaw]),
                        self.args.amcl_huber_k,
                    ),
                )
            )
            self.amcl_factors += 1

        loop_match = None
        if index > 0 and scan_pose is not None and not recovered:
            existing = self.isam.calculateEstimate()
            optimized_poses = {
                old_index: np.asarray(
                    [
                        existing.atPose2(gtsam.symbol("x", old_index)).x(),
                        existing.atPose2(gtsam.symbol("x", old_index)).y(),
                        existing.atPose2(gtsam.symbol("x", old_index)).theta(),
                    ]
                )
                for old_index in self.loop_closer.keyframes
            }
            if (
                self.loop_closer.enabled
                and index >= self.loop_closer.minimum_separation
                and index % self.loop_closer.check_interval == 0
            ):
                self.loop_checks += 1
            loop_match = self.loop_closer.find_loop(
                index,
                scan_points,
                np.asarray([scan_pose.x(), scan_pose.y(), scan_pose.theta()]),
                optimized_poses,
            )
            if loop_match is not None:
                loop_sigma_xy = float(
                    np.clip(
                        loop_match.rmse
                        / math.sqrt(max(loop_match.inlier_ratio, 0.05)),
                        self.args.loop_min_sigma_xy,
                        self.args.loop_max_sigma_xy,
                    )
                )
                graph.add(
                    gtsam.BetweenFactorPose2(
                        gtsam.symbol("x", loop_match.candidate_index),
                        key,
                        gtsam.Pose2(
                            loop_match.relative_x,
                            loop_match.relative_y,
                            loop_match.relative_yaw,
                        ),
                        dcs_noise(
                            np.asarray(
                                [
                                    loop_sigma_xy,
                                    loop_sigma_xy,
                                    math.radians(self.args.loop_sigma_yaw_deg),
                                ]
                            ),
                            self.args.loop_dcs_phi,
                        ),
                    )
                )

        self.isam.update(graph, values)
        estimate = self.isam.calculateEstimate()
        optimized = estimate.atPose2(key)
        self.loop_closer.add_keyframe(index, scan_stamp, scan_points)
        with self.state_lock:
            self.keyframe_index = index
            self.last_keyframe_odom = odom
            self.last_keyframe_imu = imu
            self.last_keyframe_scan_ns = scan_stamp
            self.latest_pose = optimized
            if loop_match is None:
                self.path_poses.append((scan_stamp, optimized))
            else:
                timestamps = [value_ns for value_ns, _ in self.path_poses] + [scan_stamp]
                self.path_poses = [
                    (value_ns, estimate.atPose2(gtsam.symbol("x", pose_index)))
                    for pose_index, value_ns in enumerate(timestamps)
                ]
                self.loop_edges.append((loop_match.candidate_index, index))
                self.loop_closures += 1
            if recovered:
                self.recovery_indices.add(index)
        if loop_match is not None:
            self.publish_loop_edges(scan_stamp)
            self.get_logger().warning(
                f"DCS loop {loop_match.candidate_index}->{index}: "
                f"inliers={loop_match.inlier_ratio:.2f}, "
                f"rmse={loop_match.rmse:.3f}m"
            )
        return optimized, loop_match

    def is_reasonable_odom_delta(self, relative: gtsam.Pose2) -> bool:
        return bool(
            math.hypot(relative.x(), relative.y()) <= self.args.maximum_odom_step
            and abs(math.degrees(relative.theta()))
            <= self.args.maximum_odom_rotation_deg
        )

    def publish_propagated_odometry(self, odom: OdomSample, source: Odometry) -> None:
        with self.state_lock:
            latest = self.latest_pose
            keyframe_odom = self.last_keyframe_odom
        if latest is None or keyframe_odom is None:
            return
        relative = keyframe_odom.pose.between(odom.pose)
        pose = latest.compose(relative) if self.is_reasonable_odom_delta(relative) else latest
        message = Odometry()
        message.header = source.header
        message.header.frame_id = self.args.map_frame
        message.child_frame_id = self.args.base_frame
        self.fill_pose(message.pose.pose, pose)
        message.twist = source.twist
        self.odom_publisher.publish(message)
        self.axes_publisher.publish(self.make_axes_marker(odom.stamp_ns, pose))
        self.publish_map_to_odom_transform(odom.stamp_ns, pose, odom.pose)

    def publish_map_to_odom_transform(
        self, value_ns: int, map_to_base: gtsam.Pose2, odom_to_base: gtsam.Pose2
    ) -> None:
        if not self.args.publish_map_to_odom_tf:
            return
        pose = map_to_base.compose(odom_to_base.inverse())
        message = TransformStamped()
        set_stamp(message.header.stamp, value_ns)
        message.header.frame_id = self.args.map_frame
        message.child_frame_id = self.args.odom_frame
        message.transform.translation.x = float(pose.x())
        message.transform.translation.y = float(pose.y())
        message.transform.rotation.z = math.sin(pose.theta() / 2.0)
        message.transform.rotation.w = math.cos(pose.theta() / 2.0)
        self.tf_broadcaster.sendTransform(message)

    @staticmethod
    def fill_pose(message_pose, pose: gtsam.Pose2) -> None:
        message_pose.position.x = float(pose.x())
        message_pose.position.y = float(pose.y())
        message_pose.orientation.z = math.sin(pose.theta() / 2.0)
        message_pose.orientation.w = math.cos(pose.theta() / 2.0)

    def make_axes_marker(self, value_ns: int, pose: gtsam.Pose2) -> Marker:
        marker = Marker()
        set_stamp(marker.header.stamp, value_ns)
        marker.header.frame_id = self.args.map_frame
        marker.ns = "indoor_pose2_dcs_axes"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.args.axis_width
        origin = Point(x=float(pose.x()), y=float(pose.y()), z=0.05)
        cosine = math.cos(pose.theta())
        sine = math.sin(pose.theta())
        endpoints = (
            Point(
                x=float(pose.x() + self.args.axis_length * cosine),
                y=float(pose.y() + self.args.axis_length * sine),
                z=0.05,
            ),
            Point(
                x=float(pose.x() - self.args.axis_length * sine),
                y=float(pose.y() + self.args.axis_length * cosine),
                z=0.05,
            ),
        )
        colors = (
            ColorRGBA(r=1.0, g=0.05, b=0.05, a=1.0),
            ColorRGBA(r=0.05, g=1.0, b=0.05, a=1.0),
        )
        for endpoint, color in zip(endpoints, colors):
            marker.points.extend((origin, endpoint))
            marker.colors.extend((color, color))
        return marker

    def publish_loop_edges(self, value_ns: int) -> None:
        with self.state_lock:
            loop_edges = list(self.loop_edges)
            path_poses = list(self.path_poses)
        marker = Marker()
        set_stamp(marker.header.stamp, value_ns)
        marker.header.frame_id = self.args.map_frame
        marker.ns = "indoor_dcs_loop_closures"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.args.loop_marker_width
        marker.color = ColorRGBA(r=1.0, g=0.55, b=0.05, a=0.9)
        for first, second in loop_edges:
            if first >= len(path_poses) or second >= len(path_poses):
                continue
            first_pose = path_poses[first][1]
            second_pose = path_poses[second][1]
            marker.points.extend(
                (
                    Point(x=first_pose.x(), y=first_pose.y(), z=0.15),
                    Point(x=second_pose.x(), y=second_pose.y(), z=0.15),
                )
            )
        self.loops_publisher.publish(marker)

    def publish_path(self) -> None:
        with self.state_lock:
            path_poses = list(self.path_poses)
            odom_only_path_poses = list(self.odom_only_path_poses)
            recovery_indices = set(self.recovery_indices)
        if path_poses:
            self.path_publisher.publish(self.make_path_message(path_poses))
            self.publish_path_markers(path_poses, recovery_indices)
        if odom_only_path_poses:
            self.odom_only_path_publisher.publish(
                self.make_path_message(odom_only_path_poses)
            )

    def publish_path_markers(
        self,
        path_poses: list[tuple[int, gtsam.Pose2]],
        recovery_indices: set[int],
    ) -> None:
        stamp = path_poses[-1][0]
        segments = Marker()
        set_stamp(segments.header.stamp, stamp)
        segments.header.frame_id = self.args.map_frame
        segments.ns = "indoor_factor_graph_segments"
        segments.id = 0
        segments.type = Marker.LINE_LIST
        segments.action = Marker.ADD
        segments.pose.orientation.w = 1.0
        segments.scale.x = 0.16
        segments.color = ColorRGBA(r=0.12, g=0.82, b=1.0, a=1.0)
        for index in range(1, len(path_poses)):
            if index in recovery_indices:
                continue
            previous = path_poses[index - 1][1]
            current = path_poses[index][1]
            segments.points.extend(
                (
                    Point(x=previous.x(), y=previous.y(), z=0.08),
                    Point(x=current.x(), y=current.y(), z=0.08),
                )
            )
        self.segments_publisher.publish(segments)

        recoveries = Marker()
        set_stamp(recoveries.header.stamp, stamp)
        recoveries.header.frame_id = self.args.map_frame
        recoveries.ns = "indoor_scan_recoveries"
        recoveries.id = 0
        recoveries.type = Marker.SPHERE_LIST
        recoveries.action = Marker.ADD
        recoveries.pose.orientation.w = 1.0
        recoveries.scale.x = 0.55
        recoveries.scale.y = 0.55
        recoveries.scale.z = 0.55
        recoveries.color = ColorRGBA(r=1.0, g=0.45, b=0.05, a=1.0)
        for index in sorted(recovery_indices):
            if index < len(path_poses):
                pose = path_poses[index][1]
                recoveries.points.append(Point(x=pose.x(), y=pose.y(), z=0.22))
        self.recovery_markers_publisher.publish(recoveries)

    def make_path_message(
        self, path_poses: list[tuple[int, gtsam.Pose2]]
    ) -> PathMsg:
        message = PathMsg()
        message.header.frame_id = self.args.map_frame
        set_stamp(message.header.stamp, path_poses[-1][0])
        for value_ns, pose in path_poses:
            stamped = PoseStamped()
            stamped.header.frame_id = self.args.map_frame
            set_stamp(stamped.header.stamp, value_ns)
            self.fill_pose(stamped.pose, pose)
            message.poses.append(stamped)
        return message

    def publish_map_once(self) -> None:
        if self.map_published:
            return
        message = OccupancyGrid()
        message.header.frame_id = self.args.map_frame
        message.info.resolution = self.map.resolution
        message.info.width = self.map.width
        message.info.height = self.map.height
        message.info.origin.position.x = self.map.origin_x
        message.info.origin.position.y = self.map.origin_y
        message.info.origin.orientation.w = 1.0
        message.data = self.map.occupancy_data().tolist()
        self.map_publisher.publish(message)
        self.map_published = True
        self.get_logger().info(f"Published fixed map on {self.args.map_topic}")

    def publish_amcl_seed(self, value_ns: int, pose: gtsam.Pose2) -> None:
        message = PoseWithCovarianceStamped()
        set_stamp(message.header.stamp, value_ns)
        message.header.frame_id = self.args.map_frame
        self.fill_pose(message.pose.pose, pose)
        message.pose.covariance[0] = self.args.prior_sigma_xy**2
        message.pose.covariance[7] = self.args.prior_sigma_xy**2
        message.pose.covariance[35] = math.radians(
            self.args.prior_sigma_yaw_deg
        ) ** 2
        self.amcl_seed_publisher.publish(message)
        self.get_logger().info(
            f"Seeded AMCL on {self.args.amcl_initialpose_topic} from x0"
        )

    def publish_status(self) -> None:
        match = self.last_match
        imu_description = "filtered odometry"
        if self.args.use_robot_imu_factor:
            imu_description += " + robot IMU"
        if self.args.use_lidar_gyro_factor:
            imu_description += " + Ouster gyro"
        payload = {
            "state": "running" if self.keyframe_index >= 0 else self.last_status,
            "model": f"Pose2 iSAM2 + DCS loops + {imu_description}",
            "keyframes": self.keyframe_index + 1,
            "odometry_only_poses": len(self.odom_only_path_poses),
            "source_hz": {
                "local_odometry": rate_hz(self.odom_rx_stamp),
                "robot_imu": rate_hz(self.imu_rx_stamp),
                "lidar_imu": rate_hz(self.lidar_imu_rx_stamp),
                "scan": rate_hz(self.scan_rx_stamp),
            },
            "arrival_hz": {
                "local_odometry": rate_hz(self.odom_rx_wall),
                "robot_imu": rate_hz(self.imu_rx_wall),
                "lidar_imu": rate_hz(self.lidar_imu_rx_wall),
                "scan": rate_hz(self.scan_rx_wall),
            },
            "processed_hz": rate_hz(self.processing_wall),
            "accepted": self.scans_accepted,
            "rejected": self.scans_rejected,
            "dropped_busy": self.scans_dropped,
            "odometry_jumps_rejected": self.odom_jumps_rejected,
            "scan_recovery": {
                "attempts": self.recovery_attempts,
                "accepted": self.recoveries,
                "consecutive_scan_rejections": self.consecutive_scan_rejections,
            },
            "amcl_recovery": {
                "enabled": self.args.enable_amcl_recovery,
                "messages": self.amcl_messages_received,
                "attempts": self.amcl_recovery_attempts,
                "rejected_hypotheses": self.amcl_recovery_rejections,
                "accepted": self.amcl_recoveries,
                "factors": self.amcl_factors,
                "last_mahalanobis_sq": self.last_amcl_mahalanobis_sq,
                "last_sigma": None
                if self.last_amcl_sigma is None
                else {
                    "x_m": self.last_amcl_sigma[0],
                    "y_m": self.last_amcl_sigma[1],
                    "yaw_deg": math.degrees(self.last_amcl_sigma[2]),
                },
            },
            "imu_factors": {
                "robot_orientation_delta": self.robot_imu_factors,
                "ouster_gyro_delta": self.lidar_gyro_factors,
            },
            "loop_closure": {
                "checks": self.loop_checks,
                "accepted": self.loop_closures,
                "robust_model": "DCS",
                "confirmation_count": self.args.loop_confirmation_count,
            },
            "last_status": self.last_status,
            "last_match": None
            if match is None
            else {
                "inlier_ratio": match.inlier_ratio,
                "mean_distance_m": match.mean_distance,
                "valid_ratio": match.valid_ratio,
                "correction_xy_m": self.last_scan_correction_xy,
                "correction_yaw_deg": self.last_scan_correction_yaw_deg,
            },
            "last_rejection_reasons": self.last_rejection_reasons,
            "rejection_reasons": self.rejection_reasons,
            "processing_ms_p50": float(np.median(self.processing_ms))
            if self.processing_ms
            else None,
            "processing_ms_p95": float(np.percentile(self.processing_ms, 95))
            if self.processing_ms
            else None,
        }
        message = String(data=json.dumps(payload, sort_keys=True))
        self.status_publisher.publish(message)
        temporary = self.args.output_status_json.with_suffix(".json.tmp")
        temporary.write_text(message.data + "\n", encoding="utf-8")
        temporary.replace(self.args.output_status_json)

    def write_output(
        self,
        value_ns: int,
        pose: gtsam.Pose2,
        status: str,
        match: MatchResult,
        loop_match: LoopMatch | None,
    ) -> None:
        self.output_writer.writerow(
            [
                value_ns,
                pose.x(),
                pose.y(),
                0.0,
                0.0,
                0.0,
                math.degrees(pose.theta()),
                status,
                match.inlier_ratio,
                match.mean_distance,
                match.valid_ratio,
                self.last_scan_correction_xy,
                self.last_scan_correction_yaw_deg,
                ";".join(self.last_rejection_reasons),
                "" if loop_match is None else loop_match.candidate_index,
                "" if loop_match is None else loop_match.rmse,
                "" if loop_match is None else loop_match.inlier_ratio,
            ]
        )
        self.output_file.flush()

    def destroy_node(self):
        self.stop_event.set()
        self.pending_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)
        self.output_file.close()
        self.odom_only_output_file.close()
        return super().destroy_node()


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--config", type=Path)
    profile_args, _ = profile_parser.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML profile containing topics and algorithm parameters",
    )
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-odom-only-csv", type=Path, required=True)
    parser.add_argument("--output-status-json", type=Path, required=True)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--odom-topic", default="/jackal_velocity_controller/odom")
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--lidar-imu-topic", default="/ouster/imu")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--initialpose-topic", default="/initialpose")
    parser.add_argument("--amcl-pose-topic", default="/indoor/amcl_pose")
    parser.add_argument(
        "--amcl-initialpose-topic", default="/indoor/amcl_initialpose"
    )
    parser.add_argument("--map-topic", default="/indoor/map")
    parser.add_argument("--output-odom-topic", default="/indoor/factor_graph/odom")
    parser.add_argument("--output-path-topic", default="/indoor/factor_graph/path")
    parser.add_argument(
        "--output-segments-topic", default="/indoor/factor_graph/segments"
    )
    parser.add_argument(
        "--output-recovery-markers-topic",
        default="/indoor/factor_graph/recovery_markers",
    )
    parser.add_argument(
        "--output-odom-only-path-topic", default="/indoor/odometry_only/path"
    )
    parser.add_argument("--output-axes-topic", default="/indoor/factor_graph/axes")
    parser.add_argument("--output-loops-topic", default="/indoor/factor_graph/loops")
    parser.add_argument("--output-status-topic", default="/indoor/factor_graph/status")
    parser.add_argument("--initial-x", type=float)
    parser.add_argument("--initial-y", type=float)
    parser.add_argument("--initial-yaw-deg", type=float)
    parser.add_argument("--wait-for-initial-pose", action="store_true")
    parser.add_argument("--keyframe-rate-hz", type=float, default=5.0)
    parser.add_argument("--odom-max-sync-sec", type=float, default=0.15)
    parser.add_argument("--imu-max-sync-sec", type=float, default=0.15)
    parser.add_argument(
        "--allow-missing-robot-imu",
        action="store_true",
        help="Use LiDAR IMU only; do not require an absent chassis IMU topic.",
    )
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--maximum-scan-range", type=float, default=30.0)
    parser.add_argument("--maximum-scan-points", type=int, default=240)
    parser.add_argument("--global-particles", type=int, default=30000)
    parser.add_argument("--global-seed", type=int, default=7)
    parser.add_argument(
        "--enable-amcl-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--amcl-recovery-rejections", type=int, default=5)
    parser.add_argument("--amcl-recovery-cooldown-keyframes", type=int, default=50)
    parser.add_argument("--amcl-recovery-retry-keyframes", type=int, default=10)
    parser.add_argument("--amcl-consistency-samples", type=int, default=3)
    parser.add_argument("--amcl-consistency-xy", type=float, default=0.75)
    parser.add_argument("--amcl-consistency-yaw-deg", type=float, default=15.0)
    parser.add_argument("--amcl-max-age-sec", type=float, default=0.75)
    parser.add_argument("--amcl-max-sigma-xy", type=float, default=1.50)
    parser.add_argument("--amcl-max-sigma-yaw-deg", type=float, default=30.0)
    parser.add_argument("--amcl-recovery-xy-window", type=float, default=2.0)
    parser.add_argument("--amcl-recovery-yaw-window-deg", type=float, default=25.0)
    parser.add_argument("--amcl-recovery-max-jump", type=float, default=8.0)
    parser.add_argument(
        "--amcl-recovery-max-yaw-difference-deg", type=float, default=100.0
    )
    parser.add_argument("--amcl-recovery-minimum-inlier-ratio", type=float, default=0.60)
    parser.add_argument("--amcl-recovery-maximum-mean-distance", type=float, default=0.35)
    parser.add_argument("--amcl-recovery-minimum-valid-ratio", type=float, default=0.85)
    parser.add_argument("--amcl-covariance-inflation", type=float, default=4.0)
    parser.add_argument("--amcl-max-mahalanobis-sq", type=float, default=25.0)
    parser.add_argument("--amcl-min-sigma-xy", type=float, default=0.25)
    parser.add_argument("--amcl-min-sigma-yaw-deg", type=float, default=5.0)
    parser.add_argument("--amcl-factor-max-sigma-xy", type=float, default=2.50)
    parser.add_argument("--amcl-factor-max-sigma-yaw-deg", type=float, default=45.0)
    parser.add_argument("--amcl-huber-k", type=float, default=1.5)
    parser.add_argument("--local-xy-window", type=float, default=0.6)
    parser.add_argument("--local-xy-step", type=float, default=0.20)
    parser.add_argument("--local-yaw-window-deg", type=float, default=6.0)
    parser.add_argument("--local-yaw-step-deg", type=float, default=1.0)
    parser.add_argument("--truncation-distance", type=float, default=1.5)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.35)
    parser.add_argument("--maximum-mean-distance", type=float, default=0.65)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.75)
    parser.add_argument("--maximum-scan-correction", type=float, default=0.75)
    parser.add_argument("--maximum-scan-rotation-deg", type=float, default=5.0)
    parser.add_argument(
        "--enable-scan-recovery", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--recovery-after-rejections", type=int, default=5)
    parser.add_argument("--recovery-check-interval", type=int, default=5)
    parser.add_argument("--recovery-particles", type=int, default=30000)
    parser.add_argument("--recovery-minimum-inlier-ratio", type=float, default=0.65)
    parser.add_argument("--recovery-maximum-mean-distance", type=float, default=0.30)
    parser.add_argument("--recovery-maximum-xy-distance", type=float, default=8.0)
    parser.add_argument("--recovery-maximum-yaw-deg", type=float, default=100.0)
    parser.add_argument("--recovery-confirmations", type=int, default=3)
    parser.add_argument("--recovery-confirmation-xy-distance", type=float, default=1.0)
    parser.add_argument("--recovery-confirmation-yaw-deg", type=float, default=15.0)
    parser.add_argument("--prior-sigma-xy", type=float, default=0.25)
    parser.add_argument("--prior-sigma-yaw-deg", type=float, default=5.0)
    parser.add_argument("--odom-sigma-xy", type=float, default=0.05)
    parser.add_argument("--odom-sigma-yaw-deg", type=float, default=2.0)
    parser.add_argument("--weak-odom-sigma-xy", type=float, default=1.0)
    parser.add_argument("--weak-odom-sigma-yaw-deg", type=float, default=30.0)
    parser.add_argument("--imu-translation-sigma", type=float, default=10.0)
    parser.add_argument("--robot-imu-delta-yaw-sigma-deg", type=float, default=2.0)
    parser.add_argument(
        "--use-robot-imu-factor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable only when odometry does not already fuse /imu/data.",
    )
    parser.add_argument("--lidar-gyro-delta-yaw-sigma-deg", type=float, default=4.0)
    parser.add_argument(
        "--use-lidar-gyro-factor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lidar-gyro-bias-z-rad-s", type=float, default=-0.0026)
    parser.add_argument("--maximum-imu-delta-yaw-deg", type=float, default=15.0)
    parser.add_argument("--scan-min-sigma-xy", type=float, default=0.08)
    parser.add_argument("--scan-max-sigma-xy", type=float, default=0.80)
    parser.add_argument("--scan-min-sigma-yaw-deg", type=float, default=1.0)
    parser.add_argument("--scan-max-sigma-yaw-deg", type=float, default=12.0)
    parser.add_argument("--scan-huber-k", type=float, default=1.5)
    parser.add_argument(
        "--enable-loop-closure", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--loop-check-interval", type=int, default=10)
    parser.add_argument("--loop-minimum-separation", type=int, default=250)
    parser.add_argument("--loop-candidate-radius", type=float, default=1.5)
    parser.add_argument("--loop-maximum-candidates", type=int, default=2)
    parser.add_argument("--loop-submap-half-width", type=int, default=6)
    parser.add_argument("--loop-voxel-size", type=float, default=0.08)
    parser.add_argument("--loop-max-correspondence-distance", type=float, default=0.30)
    parser.add_argument("--loop-minimum-inlier-ratio", type=float, default=0.70)
    parser.add_argument("--loop-maximum-rmse", type=float, default=0.18)
    parser.add_argument("--loop-minimum-correspondences", type=int, default=100)
    parser.add_argument("--loop-maximum-correction", type=float, default=0.30)
    parser.add_argument("--loop-maximum-correction-yaw-deg", type=float, default=5.0)
    parser.add_argument("--loop-confirmation-count", type=int, default=3)
    parser.add_argument("--loop-confirmation-index-tolerance", type=int, default=20)
    parser.add_argument("--loop-min-sigma-xy", type=float, default=0.05)
    parser.add_argument("--loop-max-sigma-xy", type=float, default=0.30)
    parser.add_argument("--loop-sigma-yaw-deg", type=float, default=2.0)
    parser.add_argument("--loop-dcs-phi", type=float, default=1.0)
    parser.add_argument("--maximum-odom-step", type=float, default=3.0)
    parser.add_argument("--maximum-odom-rotation-deg", type=float, default=45.0)
    parser.add_argument("--path-publish-rate-hz", type=float, default=2.0)
    parser.add_argument("--status-publish-rate-hz", type=float, default=1.0)
    parser.add_argument("--axis-length", type=float, default=0.8)
    parser.add_argument("--axis-width", type=float, default=0.07)
    parser.add_argument("--loop-marker-width", type=float, default=0.10)
    parser.add_argument(
        "--publish-map-to-odom-tf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--relinearize-threshold", type=float, default=0.01)
    parser.add_argument("--relinearize-skip", type=int, default=1)
    if profile_args.config is not None:
        try:
            apply_profile_defaults(parser, profile_args.config)
        except (OSError, ValueError, yaml.YAMLError) as error:
            parser.error(str(error))
    args, ros_args = parser.parse_known_args()
    manual = (args.initial_x, args.initial_y, args.initial_yaw_deg)
    if any(value is not None for value in manual) and not all(
        value is not None for value in manual
    ):
        parser.error("initial-x, initial-y and initial-yaw-deg must be set together")
    if args.allow_missing_robot_imu and args.use_robot_imu_factor:
        parser.error("robot IMU factor cannot be enabled when robot IMU is optional")
    if min(
        args.recovery_after_rejections,
        args.recovery_check_interval,
        args.recovery_particles,
        args.recovery_confirmations,
    ) < 1:
        parser.error("scan recovery counts and particle count must be positive")
    if min(
        args.amcl_recovery_rejections,
        args.amcl_recovery_cooldown_keyframes,
        args.amcl_recovery_retry_keyframes,
        args.amcl_consistency_samples,
    ) < 1:
        parser.error("AMCL recovery counts must be positive")
    if args.amcl_covariance_inflation < 1.0:
        parser.error("AMCL covariance inflation must be at least 1.0")
    return args, ros_args


def main() -> None:
    args, ros_args = parse_arguments()
    rclpy.init(args=ros_args)
    node = IndoorPose2FactorGraph(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
