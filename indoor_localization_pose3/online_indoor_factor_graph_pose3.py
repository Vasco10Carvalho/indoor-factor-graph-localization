#!/usr/bin/env python3

"""Incremental indoor Pose3 graph: odometry + 3D ICP + loop closures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

import gtsam
import numpy as np
import open3d as o3d
import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header, String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from pose3_icp import crop_target_map, load_map, run_icp  # noqa: E402
from profile_config import apply_profile_defaults  # noqa: E402

from pose3_common import (  # noqa: E402
    OdomSample,
    deskew_points,
    keyframe_due,
    load_pose2_aid_csv,
    map_visualization_mask,
    nearest_odom,
    nearest_pose2_aid,
    pointcloud2_xyz_time,
    pose3_from_components,
    pose3_from_odometry,
    pose_rpy_deg,
    robust_noise,
    rotation_angle_deg,
    set_stamp,
    stamp_ns,
)
from pose3_loop_closer import Pose3LoopCloser, loop_sigmas  # noqa: E402


def rate_hz(samples: deque[float]) -> float:
    if len(samples) < 2:
        return 0.0
    duration = samples[-1] - samples[0]
    return (len(samples) - 1) / duration if duration > 0.0 else 0.0


def percentile(values: deque[float], value: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), value))


class IndoorPose3FactorGraph(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("indoor_factor_graph_pose3_6d")
        self.args = args
        self.keyframe_period_ns = int(1e9 / args.keyframe_rate_hz)
        self.raw_path_period_ns = int(1e9 / args.raw_path_rate_hz)
        self.initial_pose = pose3_from_components(
            args.initial_x,
            args.initial_y,
            args.initial_z,
            args.initial_roll_deg,
            args.initial_pitch_deg,
            args.initial_yaw_deg,
        )
        self.base_to_lidar = pose3_from_components(
            args.base_to_lidar_x,
            args.base_to_lidar_y,
            args.base_to_lidar_z,
            args.base_to_lidar_roll_deg,
            args.base_to_lidar_pitch_deg,
            args.base_to_lidar_yaw_deg,
        )
        self.lidar_to_base = self.base_to_lidar.inverse()

        self.pose2_aid_samples = (
            load_pose2_aid_csv(args.pose2_aid_csv)
            if args.pose2_aid_csv is not None
            else ()
        )
        self.pose2_aid_stamps = tuple(
            sample.stamp_ns for sample in self.pose2_aid_samples
        )
        if self.pose2_aid_samples:
            self.get_logger().info(
                f"Loaded {len(self.pose2_aid_samples)} optional Pose2 aid poses "
                f"from {args.pose2_aid_csv}"
            )

        self.get_logger().info(f"Loading fixed Pose3 map: {args.map_ply}")
        (
            self.map_points,
            self.map_normals,
            self.map_colors,
            self.map_tree,
        ) = load_map(
            args.map_ply,
            args,
        )

        isam_parameters = gtsam.ISAM2Params()
        isam_parameters.setRelinearizeThreshold(args.relinearize_threshold)
        isam_parameters.relinearizeSkip = args.relinearize_skip
        self.isam = gtsam.ISAM2(isam_parameters)
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    math.radians(args.prior_sigma_roll_deg),
                    math.radians(args.prior_sigma_pitch_deg),
                    math.radians(args.prior_sigma_yaw_deg),
                    args.prior_sigma_xy,
                    args.prior_sigma_xy,
                    args.prior_sigma_z,
                ],
                dtype=float,
            )
        )
        self.odom_noise = self._diagonal_pose_noise(
            args.odom_sigma_roll_deg,
            args.odom_sigma_pitch_deg,
            args.odom_sigma_yaw_deg,
            args.odom_sigma_xy,
            args.odom_sigma_z,
        )
        self.weak_odom_noise = self._diagonal_pose_noise(
            args.weak_odom_sigma_rotation_deg,
            args.weak_odom_sigma_rotation_deg,
            args.weak_odom_sigma_rotation_deg,
            args.weak_odom_sigma_translation,
            args.weak_odom_sigma_translation,
        )
        self.pose2_aid_noise = robust_noise(
            np.asarray(
                [
                    math.radians(args.pose2_aid_sigma_roll_pitch_deg),
                    math.radians(args.pose2_aid_sigma_roll_pitch_deg),
                    math.radians(args.pose2_aid_sigma_yaw_deg),
                    args.pose2_aid_sigma_xy,
                    args.pose2_aid_sigma_xy,
                    args.pose2_aid_sigma_z,
                ],
                dtype=float,
            ),
            "huber",
            args.pose2_aid_huber_k,
        )
        self.loop_closer = Pose3LoopCloser(args, self.base_to_lidar)

        self.buffer_lock = threading.Lock()
        self.state_lock = threading.RLock()
        self.pending_lock = threading.Lock()
        self.odom_buffer: deque[OdomSample] = deque(maxlen=args.odom_buffer_size)
        self.pending_scan: PointCloud2 | None = None
        self.pending_event = threading.Event()
        self.stop_event = threading.Event()

        self.initialized = False
        self.keyframe_index = -1
        self.last_processed_scan_ns: int | None = None
        self.last_keyframe_odom: OdomSample | None = None
        self.latest_optimized_pose: gtsam.Pose3 | None = None
        self.map_to_odom: gtsam.Pose3 | None = None
        self.keyframe_stamps: list[int] = []
        self.path_poses: list[tuple[int, gtsam.Pose3]] = []
        self.raw_path_poses: list[tuple[int, gtsam.Pose3]] = []
        self.predicted_path_poses: list[tuple[int, gtsam.Pose3]] = []
        self.scan_path_poses: list[tuple[int, gtsam.Pose3]] = []
        self.pose2_aid_path_poses: list[tuple[int, gtsam.Pose3]] = []
        self.rejected_scan_poses: list[tuple[int, gtsam.Pose3]] = []
        self.raw_odom_anchor: gtsam.Pose3 | None = None
        self.last_raw_path_stamp_ns: int | None = None
        self.loop_edges: list[tuple[int, int]] = []

        self.scans_received = 0
        self.scans_processed = 0
        self.scans_accepted = 0
        self.scans_rejected = 0
        self.scans_skipped_period = 0
        self.scans_dropped_pending = 0
        self.missing_odometry = 0
        self.odom_jumps_rejected = 0
        self.scan_factors = 0
        self.pose2_aid_factors = 0
        self.pose2_aid_missing_sync = 0
        self.odom_factors = 0
        self.loop_checks = 0
        self.loop_accepted = 0
        self.loop_rejected = 0
        self.deskew_applied = 0
        self.deskew_skipped = 0
        self.last_status = "waiting_for_scan"
        self.last_deskew_status = "not_attempted"
        self.last_fitness = 0.0
        self.last_rmse = 0.0
        self.last_correspondences = 0
        self.last_correction_xyz = 0.0
        self.last_correction_xy = 0.0
        self.last_correction_z = 0.0
        self.last_correction_rotation_deg = 0.0
        self.last_processing_ms = 0.0
        self.last_icp_ms = 0.0
        self.last_isam_ms = 0.0
        self.maximum_pose_jump_m = 0.0
        self.maximum_pose_jump_deg = 0.0
        self.last_loop_correction_m = 0.0
        self.last_loop_correction_deg = 0.0

        self.odom_rx_wall: deque[float] = deque(maxlen=300)
        self.scan_rx_wall: deque[float] = deque(maxlen=120)
        self.processed_wall: deque[float] = deque(maxlen=120)
        self.processing_ms_history: deque[float] = deque(maxlen=500)
        self.icp_ms_history: deque[float] = deque(maxlen=500)
        self.isam_ms_history: deque[float] = deque(maxlen=500)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_csv_path = args.output_dir / "indoor_pose3_6d.csv"
        self.status_json_path = args.output_dir / "indoor_pose3_6d_status.json"
        self.output_file = self.output_csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self.output_writer = csv.DictWriter(
            self.output_file,
            fieldnames=(
                "stamp_ns",
                "keyframe",
                "map_x",
                "map_y",
                "map_z",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "odom_x",
                "odom_y",
                "odom_z",
                "odom_roll_deg",
                "odom_pitch_deg",
                "odom_yaw_deg",
                "predicted_x",
                "predicted_y",
                "predicted_z",
                "predicted_roll_deg",
                "predicted_pitch_deg",
                "predicted_yaw_deg",
                "scan_x",
                "scan_y",
                "scan_z",
                "scan_roll_deg",
                "scan_pitch_deg",
                "scan_yaw_deg",
                "pose2_aid_used",
                "pose2_aid_x",
                "pose2_aid_y",
                "pose2_aid_yaw_deg",
                "status",
                "icp_accepted",
                "fitness",
                "rmse",
                "correspondences",
                "correction_xyz_m",
                "correction_xy_m",
                "correction_z_m",
                "correction_rotation_deg",
                "deskew_applied",
                "deskew_status",
                "loop_accepted",
                "loop_candidate",
                "processing_ms",
                "icp_ms",
                "isam_ms",
            ),
        )
        self.output_writer.writeheader()

        sensor_qos = QoSProfile(depth=5)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        output_qos = QoSProfile(depth=10)
        output_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.create_subscription(
            Odometry,
            args.odom_topic,
            self.odom_callback,
            sensor_qos,
        )
        self.create_subscription(
            PointCloud2,
            args.lidar_topic,
            self.scan_callback,
            sensor_qos,
        )
        self.odom_publisher = self.create_publisher(
            Odometry,
            args.output_odom_topic,
            output_qos,
        )
        self.path_publisher = self.create_publisher(
            PathMsg,
            args.output_path_topic,
            latched_qos,
        )
        self.raw_path_publisher = self.create_publisher(
            PathMsg,
            args.output_raw_path_topic,
            latched_qos,
        )
        self.scan_pose_publisher = self.create_publisher(
            PoseStamped,
            args.output_scan_pose_topic,
            output_qos,
        )
        self.scan_path_publisher = self.create_publisher(
            PathMsg,
            args.output_scan_path_topic,
            latched_qos,
        )
        self.pose2_aid_path_publisher = self.create_publisher(
            PathMsg,
            args.output_pose2_aid_path_topic,
            latched_qos,
        )
        self.predicted_path_publisher = self.create_publisher(
            PathMsg,
            args.output_predicted_path_topic,
            latched_qos,
        )
        self.rejected_scan_publisher = self.create_publisher(
            Marker,
            args.output_rejected_scan_topic,
            latched_qos,
        )
        self.axes_publisher = self.create_publisher(
            Marker,
            args.output_axes_topic,
            output_qos,
        )
        self.loops_publisher = self.create_publisher(
            MarkerArray,
            args.output_loops_topic,
            latched_qos,
        )
        self.map_publisher = self.create_publisher(
            PointCloud2,
            args.output_map_topic,
            latched_qos,
        )
        self.status_publisher = self.create_publisher(
            String,
            args.output_status_topic,
            output_qos,
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.publish_base_to_lidar_static_transform()

        self.create_timer(1.0 / args.path_publish_rate_hz, self.publish_paths)
        self.create_timer(1.0 / args.status_publish_rate_hz, self.publish_status)
        self.map_timer = self.create_timer(1.0, self.publish_map_once)
        self.worker = threading.Thread(
            target=self.scan_worker,
            name="indoor_pose3_scan_worker",
            daemon=True,
        )
        self.worker.start()

        rpy = pose_rpy_deg(self.base_to_lidar)
        self.get_logger().info(
            "Indoor Pose3 graph ready (no GNSS, no AMCL): "
            f"odom={args.odom_topic}, lidar={args.lidar_topic}, "
            f"keyframes={args.keyframe_rate_hz:.2f} Hz, "
            "base_link_T_lidar="
            f"[{args.base_to_lidar_x:.3f}, {args.base_to_lidar_y:.3f}, "
            f"{args.base_to_lidar_z:.3f}, {rpy[0]:.2f}, {rpy[1]:.2f}, "
            f"{rpy[2]:.2f} deg]"
        )

    @staticmethod
    def _diagonal_pose_noise(
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
        xy: float,
        z: float,
    ):
        return gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    math.radians(roll_deg),
                    math.radians(pitch_deg),
                    math.radians(yaw_deg),
                    xy,
                    xy,
                    z,
                ],
                dtype=float,
            )
        )

    def odom_callback(self, message: Odometry) -> None:
        self.odom_rx_wall.append(time.monotonic())
        sample = OdomSample(stamp_ns(message), pose3_from_odometry(message))
        with self.buffer_lock:
            self.odom_buffer.append(sample)
        with self.state_lock:
            if self.raw_odom_anchor is None:
                self.raw_odom_anchor = sample.pose
            if (
                self.last_raw_path_stamp_ns is None
                or sample.stamp_ns - self.last_raw_path_stamp_ns
                >= self.raw_path_period_ns
            ):
                raw_aligned = self.initial_pose.compose(
                    self.raw_odom_anchor.between(sample.pose)
                )
                self.raw_path_poses.append((sample.stamp_ns, raw_aligned))
                self.last_raw_path_stamp_ns = sample.stamp_ns
            map_to_odom = self.map_to_odom
        if map_to_odom is None:
            return
        propagated = map_to_odom.compose(sample.pose)
        output = Odometry()
        output.header = message.header
        output.header.frame_id = self.args.map_frame
        output.child_frame_id = self.args.base_frame
        self.fill_pose(output.pose.pose, propagated)
        output.pose.covariance = message.pose.covariance
        output.twist = message.twist
        self.odom_publisher.publish(output)
        self.publish_map_to_odom(sample.stamp_ns, map_to_odom)

    def scan_callback(self, message: PointCloud2) -> None:
        self.scan_rx_wall.append(time.monotonic())
        self.scans_received += 1
        if self.args.lidar_frame and message.header.frame_id != self.args.lidar_frame:
            self.last_status = "rejected_lidar_frame"
            self.scans_rejected += 1
            return
        with self.pending_lock:
            if self.pending_scan is not None:
                self.scans_dropped_pending += 1
            self.pending_scan = message
            self.pending_event.set()

    def scan_worker(self) -> None:
        while not self.stop_event.is_set():
            if not self.pending_event.wait(timeout=0.2):
                continue
            with self.pending_lock:
                message = self.pending_scan
                self.pending_scan = None
                self.pending_event.clear()
            if message is None:
                continue
            try:
                self.process_scan(message)
            except Exception as exc:  # noqa: BLE001 - keep the ROS worker alive
                self.scans_rejected += 1
                self.last_status = "processing_exception"
                self.get_logger().error(f"Pose3 scan processing failed: {exc}")

    def process_scan(self, message: PointCloud2) -> None:
        started = time.monotonic()
        scan_start_ns = stamp_ns(message)
        if not keyframe_due(
            self.last_processed_scan_ns,
            scan_start_ns,
            self.keyframe_period_ns,
        ):
            self.scans_skipped_period += 1
            return

        points, point_times = pointcloud2_xyz_time(
            message,
            self.args.point_time_field,
        )
        with self.buffer_lock:
            odom_samples = tuple(self.odom_buffer)
        reference_ns = scan_start_ns
        deskew_used = False
        deskew_status = "disabled"
        if self.args.enable_deskew:
            points, reference_ns, deskew_used, deskew_status = deskew_points(
                points,
                point_times,
                reference_ns,
                odom_samples,
                self.base_to_lidar,
                time_scale=self.args.point_time_scale,
                reference=self.args.deskew_reference,
                bins=self.args.deskew_bins,
                max_odom_gap_sec=self.args.deskew_max_odom_gap_sec,
            )
            if deskew_used:
                self.deskew_applied += 1
            else:
                self.deskew_skipped += 1
        self.last_deskew_status = deskew_status

        odom = nearest_odom(
            odom_samples,
            reference_ns,
            self.args.odom_max_sync_sec,
        )
        if odom is None:
            self.missing_odometry += 1
            self.last_status = "waiting_for_synchronized_odometry"
            return

        source = self.preprocess_scan(points)
        source_points = np.asarray(source.points, dtype=float)
        with self.state_lock:
            latest = self.latest_optimized_pose
            previous_odom = self.last_keyframe_odom
        odom_delta = None
        odom_delta_valid = True
        if latest is None or previous_odom is None:
            predicted = self.initial_pose
        else:
            odom_delta = previous_odom.pose.between(odom.pose)
            odom_delta_valid = self.reasonable_odometry_delta(odom_delta)
            if odom_delta_valid:
                predicted = latest.compose(odom_delta)
            else:
                predicted = latest
                self.odom_jumps_rejected += 1

        pose2_aid = nearest_pose2_aid(
            self.pose2_aid_samples,
            self.pose2_aid_stamps,
            reference_ns,
            self.args.pose2_aid_max_sync_sec,
        )
        pose2_aid_pose = None
        if pose2_aid is not None:
            predicted_xyz = np.asarray(predicted.translation(), dtype=float)
            predicted_rpy = pose_rpy_deg(predicted)
            pose2_aid_pose = pose3_from_components(
                pose2_aid.x,
                pose2_aid.y,
                float(predicted_xyz[2]),
                float(predicted_rpy[0]),
                float(predicted_rpy[1]),
                pose2_aid.yaw_deg,
            )
        elif self.pose2_aid_samples:
            self.pose2_aid_missing_sync += 1

        scan_pose = None
        icp_candidate = None
        fitness = 0.0
        rmse = math.inf
        correspondences = 0
        correction_xyz = 0.0
        correction_xy = 0.0
        correction_z = 0.0
        correction_rotation = 0.0
        status = "rejected_too_few_source_points"
        icp_started = time.monotonic()
        if len(source.points) >= self.args.minimum_source_points:
            prior_lidar = predicted.compose(self.base_to_lidar)
            target = crop_target_map(
                self.map_points,
                self.map_normals,
                self.map_tree,
                np.asarray(prior_lidar.translation(), dtype=float),
                self.args,
            )
            if len(target.points) < self.args.min_target_points:
                status = "rejected_too_few_target_points"
            else:
                result = run_icp(
                    source,
                    target,
                    np.asarray(prior_lidar.matrix(), dtype=float),
                    self.args,
                )
                fitness = float(result.fitness)
                rmse = float(result.inlier_rmse)
                correspondences = len(result.correspondence_set)
                measured_lidar = gtsam.Pose3(
                    np.asarray(result.transformation, dtype=float)
                )
                candidate = measured_lidar.compose(self.lidar_to_base)
                icp_candidate = candidate
                correction = predicted.between(candidate)
                translation = np.asarray(correction.translation(), dtype=float)
                correction_xyz = float(np.linalg.norm(translation))
                correction_xy = float(np.linalg.norm(translation[:2]))
                correction_z = abs(float(translation[2]))
                correction_rotation = rotation_angle_deg(correction.rotation())
                if (
                    fitness < self.args.minimum_fitness
                    or rmse > self.args.maximum_rmse
                    or correspondences < self.args.minimum_correspondences
                ):
                    status = "rejected_icp_quality"
                elif (
                    correction_xyz > self.args.maximum_correction_xyz
                    or correction_xy > self.args.maximum_correction_xy
                    or correction_z > self.args.maximum_correction_z
                    or correction_rotation
                    > self.args.maximum_correction_rotation_deg
                ):
                    status = "rejected_icp_correction"
                else:
                    scan_pose = candidate
                    status = "accepted_scan_to_map"
        self.last_icp_ms = (time.monotonic() - icp_started) * 1000.0

        isam_started = time.monotonic()
        optimized, loop_match = self.update_graph(
            odom=odom,
            predicted=predicted,
            odom_delta=odom_delta,
            odom_delta_valid=odom_delta_valid,
            scan_pose=scan_pose,
            scan_fitness=fitness,
            scan_rmse=rmse,
            scan_correspondences=correspondences,
            pose2_aid_pose=pose2_aid_pose,
            source_points=source_points,
            scan_stamp_ns=reference_ns,
        )
        self.last_isam_ms = (time.monotonic() - isam_started) * 1000.0
        self.last_processing_ms = (time.monotonic() - started) * 1000.0

        self.last_processed_scan_ns = reference_ns
        self.scans_processed += 1
        self.processed_wall.append(time.monotonic())
        self.processing_ms_history.append(self.last_processing_ms)
        self.icp_ms_history.append(self.last_icp_ms)
        self.isam_ms_history.append(self.last_isam_ms)
        self.scans_accepted += int(scan_pose is not None)
        self.scans_rejected += int(scan_pose is None)
        self.last_status = (
            "accepted_loop_closure" if loop_match is not None else status
        )
        self.last_fitness = fitness
        self.last_rmse = rmse
        self.last_correspondences = correspondences
        self.last_correction_xyz = correction_xyz
        self.last_correction_xy = correction_xy
        self.last_correction_z = correction_z
        self.last_correction_rotation_deg = correction_rotation

        with self.state_lock:
            self.predicted_path_poses.append((reference_ns, predicted))
            if scan_pose is not None:
                self.scan_path_poses.append((reference_ns, scan_pose))
            elif icp_candidate is not None:
                self.rejected_scan_poses.append((reference_ns, icp_candidate))
            if pose2_aid_pose is not None:
                self.pose2_aid_path_poses.append((reference_ns, pose2_aid_pose))
        if scan_pose is not None:
            self.publish_pose(self.scan_pose_publisher, reference_ns, scan_pose)
        self.axes_publisher.publish(self.make_axes_marker(reference_ns, optimized))
        self.write_output_row(
            reference_ns,
            optimized,
            odom,
            predicted,
            icp_candidate,
            pose2_aid,
            scan_pose is not None,
            fitness,
            rmse,
            correspondences,
            correction_xyz,
            correction_xy,
            correction_z,
            correction_rotation,
            deskew_used,
            deskew_status,
            loop_match,
        )

        if self.scans_processed % 10 == 0:
            xyz = np.asarray(optimized.translation(), dtype=float)
            rpy = pose_rpy_deg(optimized)
            self.get_logger().info(
                f"k={self.keyframe_index} {self.last_status} "
                f"xyz=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}) "
                f"rpy=({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f}) "
                f"fit={fitness:.3f} rmse={rmse:.3f} corr={correspondences} "
                f"deskew={deskew_status} total={self.last_processing_ms:.1f} ms"
            )

    def preprocess_scan(self, points: np.ndarray) -> o3d.geometry.PointCloud:
        if len(points) == 0:
            return o3d.geometry.PointCloud()
        ranges = np.linalg.norm(points, axis=1)
        mask = (
            np.isfinite(points).all(axis=1)
            & (ranges >= self.args.min_range)
            & (ranges <= self.args.max_range)
            & (points[:, 2] >= self.args.source_z_min)
            & (points[:, 2] <= self.args.source_z_max)
        )
        cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(points[mask])
        )
        if self.args.scan_voxel > 0.0:
            cloud = cloud.voxel_down_sample(self.args.scan_voxel)
        if (
            self.args.max_source_points > 0
            and len(cloud.points) > self.args.max_source_points
        ):
            rng = np.random.default_rng(
                self.args.random_seed + self.scans_processed
            )
            indices = rng.choice(
                len(cloud.points),
                size=self.args.max_source_points,
                replace=False,
            )
            cloud = cloud.select_by_index(indices)
        return cloud

    def reasonable_odometry_delta(self, delta: gtsam.Pose3) -> bool:
        translation = float(
            np.linalg.norm(np.asarray(delta.translation(), dtype=float))
        )
        rotation = rotation_angle_deg(delta.rotation())
        return (
            translation <= self.args.odom_maximum_step
            and rotation <= self.args.odom_maximum_rotation_deg
        )

    def update_graph(
        self,
        *,
        odom: OdomSample,
        predicted: gtsam.Pose3,
        odom_delta: gtsam.Pose3 | None,
        odom_delta_valid: bool,
        scan_pose: gtsam.Pose3 | None,
        scan_fitness: float,
        scan_rmse: float,
        scan_correspondences: int,
        pose2_aid_pose: gtsam.Pose3 | None,
        source_points: np.ndarray,
        scan_stamp_ns: int,
    ):
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        with self.state_lock:
            index = self.keyframe_index + 1
            key = gtsam.symbol("x", index)
            if index == 0:
                graph.add(
                    gtsam.PriorFactorPose3(
                        key,
                        self.initial_pose,
                        self.prior_noise,
                    )
                )
                values.insert(key, self.initial_pose)
            else:
                previous_key = gtsam.symbol("x", index - 1)
                graph.add(
                    gtsam.BetweenFactorPose3(
                        previous_key,
                        key,
                        odom_delta
                        if odom_delta_valid and odom_delta is not None
                        else gtsam.Pose3(),
                        self.odom_noise if odom_delta_valid else self.weak_odom_noise,
                    )
                )
                self.odom_factors += 1
                values.insert(key, predicted)

            if scan_pose is not None:
                graph.add(
                    gtsam.PriorFactorPose3(
                        key,
                        scan_pose,
                        self.scan_noise(
                            scan_fitness,
                            scan_rmse,
                            scan_correspondences,
                        ),
                    )
                )
                self.scan_factors += 1

            if pose2_aid_pose is not None:
                graph.add(
                    gtsam.PriorFactorPose3(
                        key,
                        pose2_aid_pose,
                        self.pose2_aid_noise,
                    )
                )
                self.pose2_aid_factors += 1

            loop_match = None
            if index > 0 and len(source_points) >= self.args.minimum_source_points:
                existing = self.isam.calculateEstimate()
                optimized_poses = {
                    old_index: existing.atPose3(gtsam.symbol("x", old_index))
                    for old_index in self.loop_closer.keyframes
                    if existing.exists(gtsam.symbol("x", old_index))
                }
                is_check = (
                    self.args.enable_loop_closure
                    and index >= self.args.loop_minimum_separation
                    and index % self.args.loop_check_interval == 0
                )
                if is_check:
                    self.loop_checks += 1
                loop_match = self.loop_closer.find_loop(
                    index,
                    source_points,
                    scan_pose if scan_pose is not None else predicted,
                    optimized_poses,
                )
                if is_check and loop_match is None:
                    self.loop_rejected += 1
                if loop_match is not None:
                    loop_noise = robust_noise(
                        loop_sigmas(loop_match, self.args),
                        "dcs",
                        self.args.loop_dcs_phi,
                    )
                    graph.add(
                        gtsam.BetweenFactorPose3(
                            gtsam.symbol("x", loop_match.candidate_index),
                            key,
                            loop_match.relative_base_pose,
                            loop_noise,
                        )
                    )
                    self.loop_accepted += 1
                    self.loop_edges.append(
                        (loop_match.candidate_index, index)
                    )
                    self.last_loop_correction_m = (
                        loop_match.correction_translation
                    )
                    self.last_loop_correction_deg = (
                        loop_match.correction_rotation_deg
                    )

            self.isam.update(graph, values)
            estimate = self.isam.calculateEstimate()
            optimized = estimate.atPose3(key)
            if self.latest_optimized_pose is not None:
                jump = self.latest_optimized_pose.between(optimized)
                self.maximum_pose_jump_m = max(
                    self.maximum_pose_jump_m,
                    float(
                        np.linalg.norm(
                            np.asarray(jump.translation(), dtype=float)
                        )
                    ),
                )
                self.maximum_pose_jump_deg = max(
                    self.maximum_pose_jump_deg,
                    rotation_angle_deg(jump.rotation()),
                )
            self.initialized = True
            self.keyframe_index = index
            self.last_keyframe_odom = odom
            self.latest_optimized_pose = optimized
            self.map_to_odom = optimized.compose(odom.pose.inverse())
            self.keyframe_stamps.append(scan_stamp_ns)
            self.path_poses = [
                (
                    self.keyframe_stamps[old_index],
                    estimate.atPose3(gtsam.symbol("x", old_index)),
                )
                for old_index in range(index + 1)
            ]
            self.loop_closer.add_keyframe(
                index,
                scan_stamp_ns,
                source_points,
            )
        return optimized, loop_match

    def scan_noise(
        self,
        fitness: float,
        rmse: float,
        correspondences: int,
    ):
        quality = max(fitness, 0.05)
        correspondence_scale = math.sqrt(
            max(self.args.minimum_correspondences, 1)
            / max(correspondences, self.args.minimum_correspondences)
        )
        sigma_xy = float(
            np.clip(
                rmse / math.sqrt(quality) * correspondence_scale,
                self.args.scan_min_sigma_xy,
                self.args.scan_max_sigma_xy,
            )
        )
        sigma_z = float(
            np.clip(
                1.25 * rmse / math.sqrt(quality) * correspondence_scale,
                self.args.scan_min_sigma_z,
                self.args.scan_max_sigma_z,
            )
        )
        sigma_rp = float(
            np.clip(
                1.0 + 8.0 * (1.0 - min(fitness, 1.0)) + 5.0 * rmse,
                self.args.scan_min_sigma_roll_pitch_deg,
                self.args.scan_max_sigma_roll_pitch_deg,
            )
        )
        sigma_yaw = float(
            np.clip(
                1.5 + 10.0 * (1.0 - min(fitness, 1.0)) + 6.0 * rmse,
                self.args.scan_min_sigma_yaw_deg,
                self.args.scan_max_sigma_yaw_deg,
            )
        )
        return robust_noise(
            np.asarray(
                [
                    math.radians(sigma_rp),
                    math.radians(sigma_rp),
                    math.radians(sigma_yaw),
                    sigma_xy,
                    sigma_xy,
                    sigma_z,
                ],
                dtype=float,
            ),
            "huber",
            self.args.scan_huber_k,
        )

    @staticmethod
    def fill_pose(message_pose, pose: gtsam.Pose3) -> None:
        translation = pose.translation()
        quaternion = pose.rotation().toQuaternion()
        message_pose.position.x = float(translation[0])
        message_pose.position.y = float(translation[1])
        message_pose.position.z = float(translation[2])
        message_pose.orientation.x = float(quaternion.x())
        message_pose.orientation.y = float(quaternion.y())
        message_pose.orientation.z = float(quaternion.z())
        message_pose.orientation.w = float(quaternion.w())

    def publish_pose(self, publisher, pose_stamp_ns: int, pose: gtsam.Pose3) -> None:
        message = PoseStamped()
        message.header.frame_id = self.args.map_frame
        set_stamp(message.header.stamp, pose_stamp_ns)
        self.fill_pose(message.pose, pose)
        publisher.publish(message)

    def publish_map_to_odom(
        self,
        pose_stamp_ns: int,
        map_to_odom: gtsam.Pose3,
    ) -> None:
        message = TransformStamped()
        message.header.frame_id = self.args.map_frame
        message.child_frame_id = self.args.odom_frame
        set_stamp(message.header.stamp, pose_stamp_ns)
        translation = map_to_odom.translation()
        quaternion = map_to_odom.rotation().toQuaternion()
        message.transform.translation.x = float(translation[0])
        message.transform.translation.y = float(translation[1])
        message.transform.translation.z = float(translation[2])
        message.transform.rotation.x = float(quaternion.x())
        message.transform.rotation.y = float(quaternion.y())
        message.transform.rotation.z = float(quaternion.z())
        message.transform.rotation.w = float(quaternion.w())
        self.tf_broadcaster.sendTransform(message)

    def publish_base_to_lidar_static_transform(self) -> None:
        message = TransformStamped()
        message.header.frame_id = self.args.base_frame
        message.child_frame_id = self.args.lidar_frame
        message.header.stamp = self.get_clock().now().to_msg()
        translation = self.base_to_lidar.translation()
        quaternion = self.base_to_lidar.rotation().toQuaternion()
        message.transform.translation.x = float(translation[0])
        message.transform.translation.y = float(translation[1])
        message.transform.translation.z = float(translation[2])
        message.transform.rotation.x = float(quaternion.x())
        message.transform.rotation.y = float(quaternion.y())
        message.transform.rotation.z = float(quaternion.z())
        message.transform.rotation.w = float(quaternion.w())
        self.static_tf_broadcaster.sendTransform(message)

    def make_axes_marker(self, pose_stamp_ns: int, pose: gtsam.Pose3) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.args.map_frame
        set_stamp(marker.header.stamp, pose_stamp_ns)
        marker.ns = "indoor_pose3_6d_axes"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.args.axis_width
        origin_xyz = np.asarray(pose.translation(), dtype=float)
        rotation = np.asarray(pose.rotation().matrix(), dtype=float)
        colors = (
            ColorRGBA(r=1.0, g=0.05, b=0.05, a=1.0),
            ColorRGBA(r=0.05, g=1.0, b=0.05, a=1.0),
            ColorRGBA(r=0.05, g=0.25, b=1.0, a=1.0),
        )
        for axis, color in enumerate(colors):
            endpoint = origin_xyz + self.args.axis_length * rotation[:, axis]
            marker.points.extend(
                (
                    Point(x=float(origin_xyz[0]), y=float(origin_xyz[1]), z=float(origin_xyz[2])),
                    Point(x=float(endpoint[0]), y=float(endpoint[1]), z=float(endpoint[2])),
                )
            )
            marker.colors.extend((color, color))
        return marker

    def publish_paths(self) -> None:
        with self.state_lock:
            optimized = tuple(self.path_poses)
            raw = tuple(self.raw_path_poses)
            predicted = tuple(self.predicted_path_poses)
            scan = tuple(self.scan_path_poses)
            pose2_aid = tuple(self.pose2_aid_path_poses)
            rejected = tuple(self.rejected_scan_poses)
            loops = tuple(self.loop_edges)
        if optimized:
            self.path_publisher.publish(self.make_path(optimized))
        if raw:
            self.raw_path_publisher.publish(self.make_path(raw))
        if predicted:
            self.predicted_path_publisher.publish(self.make_path(predicted))
        if scan:
            self.scan_path_publisher.publish(self.make_path(scan))
        if pose2_aid:
            self.pose2_aid_path_publisher.publish(self.make_path(pose2_aid))
        if rejected:
            self.publish_rejected_scan_markers(rejected)
        if optimized:
            self.publish_loop_markers(optimized, loops)

    def publish_rejected_scan_markers(
        self,
        poses: tuple[tuple[int, gtsam.Pose3], ...],
    ) -> None:
        marker = Marker()
        marker.header.frame_id = self.args.map_frame
        set_stamp(marker.header.stamp, poses[-1][0])
        marker.ns = "indoor_pose3_rejected_scan_matches"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color = ColorRGBA(r=0.95, g=0.12, b=0.25, a=0.95)
        marker.points = [
            Point(
                x=float(pose.translation()[0]),
                y=float(pose.translation()[1]),
                z=float(pose.translation()[2]),
            )
            for _, pose in poses
        ]
        self.rejected_scan_publisher.publish(marker)

    def make_path(self, poses: tuple[tuple[int, gtsam.Pose3], ...]) -> PathMsg:
        message = PathMsg()
        message.header.frame_id = self.args.map_frame
        set_stamp(message.header.stamp, poses[-1][0])
        for pose_stamp_ns, pose in poses:
            stamped = PoseStamped()
            stamped.header.frame_id = self.args.map_frame
            set_stamp(stamped.header.stamp, pose_stamp_ns)
            self.fill_pose(stamped.pose, pose)
            message.poses.append(stamped)
        return message

    def publish_loop_markers(
        self,
        poses: tuple[tuple[int, gtsam.Pose3], ...],
        loops: tuple[tuple[int, int], ...],
    ) -> None:
        marker = Marker()
        marker.header.frame_id = self.args.map_frame
        set_stamp(marker.header.stamp, poses[-1][0])
        marker.ns = "indoor_pose3_6d_loops"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.args.loop_marker_width
        marker.color = ColorRGBA(r=1.0, g=0.1, b=0.9, a=1.0)
        for first, second in loops:
            if first >= len(poses) or second >= len(poses):
                continue
            for index in (first, second):
                xyz = poses[index][1].translation()
                marker.points.append(
                    Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
                )
        array = MarkerArray()
        array.markers.append(marker)
        self.loops_publisher.publish(array)

    def publish_map_once(self) -> None:
        header = Header()
        header.frame_id = self.args.map_frame
        header.stamp = self.get_clock().now().to_msg()
        points = self.map_points
        colors = self.map_colors
        visible = map_visualization_mask(
            points,
            self.args.map_publish_z_min,
            self.args.map_publish_z_max,
        )
        points = points[visible]
        if colors is not None:
            colors = colors[visible]
        maximum = self.args.map_publish_max_points
        if maximum > 0 and len(points) > maximum:
            rng = np.random.default_rng(self.args.random_seed)
            indices = rng.choice(len(points), size=maximum, replace=False)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]
        if colors is None:
            message = point_cloud2.create_cloud_xyz32(
                header,
                np.asarray(points, dtype=np.float32),
            )
        else:
            fields = [
                PointField(
                    name="x", offset=0, datatype=PointField.FLOAT32, count=1
                ),
                PointField(
                    name="y", offset=4, datatype=PointField.FLOAT32, count=1
                ),
                PointField(
                    name="z", offset=8, datatype=PointField.FLOAT32, count=1
                ),
                PointField(
                    name="rgb", offset=12, datatype=PointField.UINT32, count=1
                ),
            ]
            cloud = np.empty(
                len(points),
                dtype=point_cloud2.dtype_from_fields(fields),
            )
            cloud["x"] = points[:, 0].astype(np.float32)
            cloud["y"] = points[:, 1].astype(np.float32)
            cloud["z"] = points[:, 2].astype(np.float32)
            rgb = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint32)
            cloud["rgb"] = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
            message = point_cloud2.create_cloud(header, fields, cloud)
        self.map_publisher.publish(message)
        self.map_timer.cancel()
        self.get_logger().info(
            f"Published {len(points)} visualization-only map points "
            f"({'RGB' if colors is not None else 'XYZ'}) on "
            f"{self.args.output_map_topic}; the ICP map remains unchanged"
        )

    def status_payload(self) -> dict:
        with self.state_lock:
            pose = self.latest_optimized_pose or self.initial_pose
            keyframes = self.keyframe_index + 1
        xyz = np.asarray(pose.translation(), dtype=float)
        rpy = pose_rpy_deg(pose)
        accepted_percentage = (
            100.0 * self.scans_accepted / self.scans_processed
            if self.scans_processed
            else 0.0
        )
        return {
            "state": "running" if self.initialized else "waiting_for_first_keyframe",
            "model": "Pose3",
            "inputs": {
                "odometry_topic": self.args.odom_topic,
                "lidar_topic": self.args.lidar_topic,
                "gnss": False,
                "amcl": False,
                "direct_imu_factor": False,
                "pose2_aid_csv": (
                    str(self.args.pose2_aid_csv)
                    if self.args.pose2_aid_csv is not None
                    else None
                ),
            },
            "rates_hz": {
                "odometry": rate_hz(self.odom_rx_wall),
                "lidar": rate_hz(self.scan_rx_wall),
                "processed": rate_hz(self.processed_wall),
                "configured_keyframes": self.args.keyframe_rate_hz,
            },
            "keyframes": keyframes,
            "pose": {
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "roll_deg": float(rpy[0]),
                "pitch_deg": float(rpy[1]),
                "yaw_deg": float(rpy[2]),
            },
            "scan_to_map": {
                "status": self.last_status,
                "received": self.scans_received,
                "processed": self.scans_processed,
                "accepted": self.scans_accepted,
                "rejected": self.scans_rejected,
                "accepted_percentage": accepted_percentage,
                "fitness": self.last_fitness,
                "rmse": None if not math.isfinite(self.last_rmse) else self.last_rmse,
                "correspondences": self.last_correspondences,
                "correction_xyz_m": self.last_correction_xyz,
                "correction_xy_m": self.last_correction_xy,
                "correction_z_m": self.last_correction_z,
                "correction_rotation_deg": self.last_correction_rotation_deg,
                "factors": self.scan_factors,
            },
            "odometry": {
                "factors": self.odom_factors,
                "missing_sync": self.missing_odometry,
                "jumps_rejected": self.odom_jumps_rejected,
            },
            "pose2_aid": {
                "enabled": bool(self.pose2_aid_samples),
                "available_poses": len(self.pose2_aid_samples),
                "factors": self.pose2_aid_factors,
                "missing_sync": self.pose2_aid_missing_sync,
                "max_sync_sec": self.args.pose2_aid_max_sync_sec,
            },
            "deskew": {
                "enabled": self.args.enable_deskew,
                "last_status": self.last_deskew_status,
                "applied": self.deskew_applied,
                "skipped": self.deskew_skipped,
            },
            "loop_closure": {
                "enabled": self.args.enable_loop_closure,
                "checks": self.loop_checks,
                "accepted": self.loop_accepted,
                "rejected": self.loop_rejected,
                "last_correction_m": self.last_loop_correction_m,
                "last_correction_deg": self.last_loop_correction_deg,
            },
            "runtime": {
                "dropped_pending_scans": self.scans_dropped_pending,
                "skipped_by_period": self.scans_skipped_period,
                "processing_ms_p50": percentile(self.processing_ms_history, 50),
                "processing_ms_p95": percentile(self.processing_ms_history, 95),
                "icp_ms_p50": percentile(self.icp_ms_history, 50),
                "icp_ms_p95": percentile(self.icp_ms_history, 95),
                "isam_ms_p50": percentile(self.isam_ms_history, 50),
                "isam_ms_p95": percentile(self.isam_ms_history, 95),
                "maximum_pose_jump_m": self.maximum_pose_jump_m,
                "maximum_pose_jump_deg": self.maximum_pose_jump_deg,
            },
        }

    def publish_status(self) -> None:
        payload = self.status_payload()
        encoded = json.dumps(payload, sort_keys=True)
        self.status_publisher.publish(String(data=encoded))
        temporary = self.status_json_path.with_suffix(".json.tmp")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        temporary.replace(self.status_json_path)

    def write_output_row(
        self,
        pose_stamp_ns: int,
        optimized: gtsam.Pose3,
        odom: OdomSample,
        predicted: gtsam.Pose3,
        scan_candidate: gtsam.Pose3 | None,
        pose2_aid,
        scan_accepted: bool,
        fitness: float,
        rmse: float,
        correspondences: int,
        correction_xyz: float,
        correction_xy: float,
        correction_z: float,
        correction_rotation: float,
        deskew_used: bool,
        deskew_status: str,
        loop_match,
    ) -> None:
        xyz = np.asarray(optimized.translation(), dtype=float)
        odom_xyz = np.asarray(odom.pose.translation(), dtype=float)
        odom_rpy = pose_rpy_deg(odom.pose)
        rpy = pose_rpy_deg(optimized)
        predicted_xyz = np.asarray(predicted.translation(), dtype=float)
        predicted_rpy = pose_rpy_deg(predicted)
        if scan_candidate is None:
            scan_values: list[str] = [""] * 6
        else:
            scan_xyz = np.asarray(scan_candidate.translation(), dtype=float)
            scan_rpy = pose_rpy_deg(scan_candidate)
            scan_values = [
                f"{scan_xyz[0]:.12f}",
                f"{scan_xyz[1]:.12f}",
                f"{scan_xyz[2]:.12f}",
                f"{scan_rpy[0]:.9f}",
                f"{scan_rpy[1]:.9f}",
                f"{scan_rpy[2]:.9f}",
            ]
        self.output_writer.writerow(
            {
                "stamp_ns": pose_stamp_ns,
                "keyframe": self.keyframe_index,
                "map_x": f"{xyz[0]:.12f}",
                "map_y": f"{xyz[1]:.12f}",
                "map_z": f"{xyz[2]:.12f}",
                "roll_deg": f"{rpy[0]:.9f}",
                "pitch_deg": f"{rpy[1]:.9f}",
                "yaw_deg": f"{rpy[2]:.9f}",
                "odom_x": f"{odom_xyz[0]:.12f}",
                "odom_y": f"{odom_xyz[1]:.12f}",
                "odom_z": f"{odom_xyz[2]:.12f}",
                "odom_roll_deg": f"{odom_rpy[0]:.9f}",
                "odom_pitch_deg": f"{odom_rpy[1]:.9f}",
                "odom_yaw_deg": f"{odom_rpy[2]:.9f}",
                "predicted_x": f"{predicted_xyz[0]:.12f}",
                "predicted_y": f"{predicted_xyz[1]:.12f}",
                "predicted_z": f"{predicted_xyz[2]:.12f}",
                "predicted_roll_deg": f"{predicted_rpy[0]:.9f}",
                "predicted_pitch_deg": f"{predicted_rpy[1]:.9f}",
                "predicted_yaw_deg": f"{predicted_rpy[2]:.9f}",
                "scan_x": scan_values[0],
                "scan_y": scan_values[1],
                "scan_z": scan_values[2],
                "scan_roll_deg": scan_values[3],
                "scan_pitch_deg": scan_values[4],
                "scan_yaw_deg": scan_values[5],
                "pose2_aid_used": int(pose2_aid is not None),
                "pose2_aid_x": (
                    f"{pose2_aid.x:.12f}" if pose2_aid is not None else ""
                ),
                "pose2_aid_y": (
                    f"{pose2_aid.y:.12f}" if pose2_aid is not None else ""
                ),
                "pose2_aid_yaw_deg": (
                    f"{pose2_aid.yaw_deg:.9f}" if pose2_aid is not None else ""
                ),
                "status": self.last_status,
                "icp_accepted": int(scan_accepted),
                "fitness": f"{fitness:.9f}",
                "rmse": "" if not math.isfinite(rmse) else f"{rmse:.9f}",
                "correspondences": correspondences,
                "correction_xyz_m": f"{correction_xyz:.9f}",
                "correction_xy_m": f"{correction_xy:.9f}",
                "correction_z_m": f"{correction_z:.9f}",
                "correction_rotation_deg": f"{correction_rotation:.9f}",
                "deskew_applied": int(deskew_used),
                "deskew_status": deskew_status,
                "loop_accepted": int(loop_match is not None),
                "loop_candidate": (
                    loop_match.candidate_index if loop_match is not None else ""
                ),
                "processing_ms": f"{self.last_processing_ms:.6f}",
                "icp_ms": f"{self.last_icp_ms:.6f}",
                "isam_ms": f"{self.last_isam_ms:.6f}",
            }
        )
        self.output_file.flush()

    def stop(self) -> None:
        self.stop_event.set()
        self.pending_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)
        self.output_file.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML profile containing topics and algorithm parameters",
    )
    parser.add_argument("--map-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pose2-aid-csv",
        type=Path,
        help="Optional timestamped Pose2 trajectory used only as x/y/yaw factors",
    )
    parser.add_argument("--initial-x", type=float, required=True)
    parser.add_argument("--initial-y", type=float, required=True)
    parser.add_argument("--initial-z", type=float, required=True)
    parser.add_argument("--initial-roll-deg", type=float, required=True)
    parser.add_argument("--initial-pitch-deg", type=float, required=True)
    parser.add_argument("--initial-yaw-deg", type=float, required=True)
    parser.add_argument("--base-to-lidar-x", type=float, required=True)
    parser.add_argument("--base-to-lidar-y", type=float, required=True)
    parser.add_argument("--base-to-lidar-z", type=float, required=True)
    parser.add_argument("--base-to-lidar-roll-deg", type=float, required=True)
    parser.add_argument("--base-to-lidar-pitch-deg", type=float, required=True)
    parser.add_argument("--base-to-lidar-yaw-deg", type=float, required=True)

    parser.add_argument("--odom-topic", default="/odometry/filtered")
    parser.add_argument("--lidar-topic", default="/ouster/points")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--lidar-frame", default="os_lidar")
    parser.add_argument("--output-odom-topic", default="/indoor_pose3/odometry")
    parser.add_argument("--output-path-topic", default="/indoor_pose3/path")
    parser.add_argument("--output-raw-path-topic", default="/indoor_pose3/odometry_path")
    parser.add_argument("--output-scan-pose-topic", default="/indoor_pose3/scan_to_map")
    parser.add_argument("--output-scan-path-topic", default="/indoor_pose3/scan_to_map_path")
    parser.add_argument(
        "--output-pose2-aid-path-topic",
        default="/indoor_pose3/pose2_aid_path",
    )
    parser.add_argument("--output-predicted-path-topic", default="/indoor_pose3/predicted_path")
    parser.add_argument("--output-rejected-scan-topic", default="/indoor_pose3/rejected_scan_markers")
    parser.add_argument("--output-axes-topic", default="/indoor_pose3/axes")
    parser.add_argument("--output-loops-topic", default="/indoor_pose3/loops")
    parser.add_argument("--output-map-topic", default="/indoor_pose3/map_points")
    parser.add_argument("--output-status-topic", default="/indoor_pose3/status")

    parser.add_argument("--keyframe-rate-hz", type=float, default=2.0)
    parser.add_argument("--raw-path-rate-hz", type=float, default=5.0)
    parser.add_argument("--path-publish-rate-hz", type=float, default=1.0)
    parser.add_argument("--status-publish-rate-hz", type=float, default=1.0)
    parser.add_argument("--odom-buffer-size", type=int, default=4000)
    parser.add_argument("--odom-max-sync-sec", type=float, default=0.04)

    parser.add_argument("--enable-deskew", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--point-time-field", default="t")
    parser.add_argument("--point-time-scale", type=float, default=1e-9)
    parser.add_argument("--deskew-reference", choices=("start", "end"), default="start")
    parser.add_argument("--deskew-bins", type=int, default=64)
    parser.add_argument("--deskew-max-odom-gap-sec", type=float, default=0.04)

    parser.add_argument("--prior-sigma-xy", type=float, default=0.25)
    parser.add_argument("--prior-sigma-z", type=float, default=0.15)
    parser.add_argument("--prior-sigma-roll-deg", type=float, default=3.0)
    parser.add_argument("--prior-sigma-pitch-deg", type=float, default=3.0)
    parser.add_argument("--prior-sigma-yaw-deg", type=float, default=8.0)
    parser.add_argument("--odom-sigma-xy", type=float, default=0.08)
    parser.add_argument("--odom-sigma-z", type=float, default=0.10)
    parser.add_argument("--odom-sigma-roll-deg", type=float, default=1.5)
    parser.add_argument("--odom-sigma-pitch-deg", type=float, default=1.5)
    parser.add_argument("--odom-sigma-yaw-deg", type=float, default=2.5)
    parser.add_argument("--weak-odom-sigma-translation", type=float, default=1.0)
    parser.add_argument("--weak-odom-sigma-rotation-deg", type=float, default=30.0)
    parser.add_argument("--odom-maximum-step", type=float, default=2.0)
    parser.add_argument("--odom-maximum-rotation-deg", type=float, default=45.0)
    parser.add_argument("--relinearize-threshold", type=float, default=0.05)
    parser.add_argument("--relinearize-skip", type=int, default=1)

    parser.add_argument("--pose2-aid-max-sync-sec", type=float, default=0.25)
    parser.add_argument("--pose2-aid-sigma-xy", type=float, default=0.12)
    parser.add_argument("--pose2-aid-sigma-yaw-deg", type=float, default=2.5)
    parser.add_argument("--pose2-aid-sigma-z", type=float, default=100.0)
    parser.add_argument(
        "--pose2-aid-sigma-roll-pitch-deg", type=float, default=90.0
    )
    parser.add_argument("--pose2-aid-huber-k", type=float, default=1.5)

    parser.add_argument("--map-voxel", type=float, default=0.10)
    parser.add_argument("--map-z-min", type=float, default=-0.5)
    parser.add_argument("--map-z-max", type=float, default=3.2)
    parser.add_argument("--map-normal-radius", type=float, default=0.45)
    parser.add_argument("--map-normal-max-nn", type=int, default=40)
    parser.add_argument("--map-publish-z-min", type=float)
    parser.add_argument("--map-publish-z-max", type=float)
    parser.add_argument("--map-publish-max-points", type=int, default=300000)
    parser.add_argument("--scan-voxel", type=float, default=0.15)
    parser.add_argument("--min-range", type=float, default=0.8)
    parser.add_argument("--max-range", type=float, default=30.0)
    parser.add_argument("--source-z-min", type=float, default=-1.2)
    parser.add_argument("--source-z-max", type=float, default=2.5)
    parser.add_argument("--max-source-points", type=int, default=12000)
    parser.add_argument("--minimum-source-points", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--crop-radius", type=float, default=16.0)
    parser.add_argument("--crop-z-margin-below", type=float, default=2.0)
    parser.add_argument("--crop-z-margin-above", type=float, default=3.0)
    parser.add_argument("--max-target-points", type=int, default=120000)
    parser.add_argument("--min-target-points", type=int, default=3000)
    parser.add_argument("--icp-method", choices=("point_to_plane",), default="point_to_plane")
    parser.add_argument("--coarse-max-correspondence", type=float, default=1.20)
    parser.add_argument("--fine-max-correspondence", type=float, default=0.35)
    parser.add_argument("--icp-iterations", type=int, default=35)
    parser.add_argument("--minimum-fitness", type=float, default=0.55)
    parser.add_argument("--maximum-rmse", type=float, default=0.20)
    parser.add_argument("--minimum-correspondences", type=int, default=1200)
    parser.add_argument("--maximum-correction-xyz", type=float, default=0.80)
    parser.add_argument("--maximum-correction-xy", type=float, default=0.75)
    parser.add_argument("--maximum-correction-z", type=float, default=0.35)
    parser.add_argument("--maximum-correction-rotation-deg", type=float, default=8.0)
    parser.add_argument("--scan-min-sigma-xy", type=float, default=0.04)
    parser.add_argument("--scan-max-sigma-xy", type=float, default=0.50)
    parser.add_argument("--scan-min-sigma-z", type=float, default=0.05)
    parser.add_argument("--scan-max-sigma-z", type=float, default=0.60)
    parser.add_argument("--scan-min-sigma-roll-pitch-deg", type=float, default=0.8)
    parser.add_argument("--scan-max-sigma-roll-pitch-deg", type=float, default=8.0)
    parser.add_argument("--scan-min-sigma-yaw-deg", type=float, default=1.0)
    parser.add_argument("--scan-max-sigma-yaw-deg", type=float, default=10.0)
    parser.add_argument("--scan-huber-k", type=float, default=1.5)

    parser.add_argument("--enable-loop-closure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loop-check-interval", type=int, default=10)
    parser.add_argument("--loop-minimum-separation", type=int, default=180)
    parser.add_argument("--loop-candidate-radius", type=float, default=2.0)
    parser.add_argument("--loop-maximum-candidates", type=int, default=2)
    parser.add_argument("--loop-submap-half-width", type=int, default=4)
    parser.add_argument("--loop-voxel", type=float, default=0.25)
    parser.add_argument("--loop-max-points-per-keyframe", type=int, default=3500)
    parser.add_argument("--loop-normal-radius", type=float, default=0.75)
    parser.add_argument("--loop-normal-max-nn", type=int, default=35)
    parser.add_argument("--loop-max-correspondence", type=float, default=0.40)
    parser.add_argument("--loop-icp-iterations", type=int, default=40)
    parser.add_argument("--loop-minimum-fitness", type=float, default=0.65)
    parser.add_argument("--loop-maximum-rmse", type=float, default=0.16)
    parser.add_argument("--loop-minimum-correspondences", type=int, default=700)
    parser.add_argument("--loop-maximum-correction", type=float, default=0.45)
    parser.add_argument("--loop-maximum-rotation-deg", type=float, default=6.0)
    parser.add_argument("--loop-confirmation-count", type=int, default=2)
    parser.add_argument("--loop-confirmation-index-tolerance", type=int, default=25)
    parser.add_argument("--loop-min-sigma-translation", type=float, default=0.05)
    parser.add_argument("--loop-max-sigma-translation", type=float, default=0.35)
    parser.add_argument("--loop-min-sigma-rotation-deg", type=float, default=1.0)
    parser.add_argument("--loop-max-sigma-rotation-deg", type=float, default=8.0)
    parser.add_argument("--loop-dcs-phi", type=float, default=1.0)

    parser.add_argument("--axis-length", type=float, default=0.8)
    parser.add_argument("--axis-width", type=float, default=0.05)
    parser.add_argument("--loop-marker-width", type=float, default=0.08)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.map_ply.is_file():
        parser.error(f"map PLY does not exist: {args.map_ply}")
    if args.keyframe_rate_hz <= 0.0 or args.raw_path_rate_hz <= 0.0:
        parser.error("keyframe and raw path rates must be positive")
    if args.path_publish_rate_hz <= 0.0 or args.status_publish_rate_hz <= 0.0:
        parser.error("publish rates must be positive")
    if args.minimum_correspondences <= 0:
        parser.error("minimum correspondences must be positive")
    if args.pose2_aid_csv is not None and not args.pose2_aid_csv.is_file():
        parser.error(f"Pose2 aid CSV does not exist: {args.pose2_aid_csv}")
    if args.pose2_aid_max_sync_sec <= 0.0:
        parser.error("pose2-aid-max-sync-sec must be positive")
    if (
        args.map_publish_z_min is not None
        and args.map_publish_z_max is not None
        and args.map_publish_z_min > args.map_publish_z_max
    ):
        parser.error("map-publish-z-min cannot exceed map-publish-z-max")


def main() -> int:
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--config", type=Path)
    profile_args, _ = profile_parser.parse_known_args()
    parser = build_parser()
    if profile_args.config is not None:
        apply_profile_defaults(parser, profile_args.config)
    args, ros_args = parser.parse_known_args()
    validate_args(parser, args)
    rclpy.init(args=ros_args)
    node = IndoorPose3FactorGraph(args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
