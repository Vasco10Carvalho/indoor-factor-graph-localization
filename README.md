# Indoor Factor Graph Localization

Portable ROS 2 Humble localization on a known 2D occupancy map. The system
uses a GTSAM `Pose2` factor graph to combine:

- relative wheel/filtered odometry;
- IMU synchronization and optional yaw factors;
- robust 2D LiDAR scan-to-map matching;
- confirmed global scan recovery after tracking loss;
- scan-to-old-submap loop closures with DCS robust noise.

It is a **localization** pipeline: it publishes the supplied map and never
creates or modifies it. GNSS and AMCL are not required.

## Why there is no `.env`

Machine-specific paths and the initial pose are command-line arguments.
Algorithm parameters and ROS topic names live in the versioned
[`configs/hospital.yaml`](configs/hospital.yaml) profile. A user can copy that
YAML, change only the topics or tuning, and keep bags/maps outside Git.

## Requirements

- Linux with Docker Engine;
- an X11/XWayland graphical session to run RViz;
- a Nav2-compatible occupancy map: `map.yaml` plus its relative `.pgm`/`.png`;
- ROS 2 data containing:
  - `nav_msgs/msg/Odometry`;
  - `sensor_msgs/msg/LaserScan`;
  - `sensor_msgs/msg/Imu` unless `allow_missing_robot_imu: true` is configured;
  - optional LiDAR IMU.

This pipeline consumes a planar `LaserScan`, not `PointCloud2` directly. A 3D
LiDAR cloud must first be projected to `/scan` using the correct LiDAR-to-base
extrinsic and a suitable height slice.

## Build and validate

```bash
git clone https://github.com/Vasco10Carvalho/indoor-factor-graph-localization.git
cd indoor-factor-graph-localization

./pipeline.sh build
./pipeline.sh check
```

No ROS installation is required on the host. GTSAM, ROS 2 Humble, rosbag2 and
RViz are inside the image.

## Inspect a bag

```bash
./pipeline.sh inspect --bag /absolute/path/to/rosbag2
```

Confirm the message types and adapt a copy of the profile when topic names are
different:

```bash
cp configs/hospital.yaml configs/my_robot.yaml
```

## Run with a known initial pose

All terminals must use the same ROS domain ID. Angles are in degrees and map
coordinates are in metres.

Terminal 1, localization:

```bash
./pipeline.sh run \
  --map /absolute/path/to/map.yaml \
  --output /absolute/path/to/output \
  --config configs/hospital.yaml \
  --initial-pose 8.70 0.09 0.8 \
  --domain-id 42
```

Terminal 2, rosbag replay:

```bash
./pipeline.sh play \
  --bag /absolute/path/to/rosbag2 \
  --rate 1.0 \
  --start-offset 0.0 \
  --domain-id 42
```

Terminal 3, RViz:

```bash
./pipeline.sh rviz --domain-id 42
```

Terminal 4, optional diagnostics:

```bash
./pipeline.sh monitor --domain-id 42
```

Use `map` as RViz Fixed Frame. The principal displays are already configured:

- `/indoor/map`;
- `/indoor/factor_graph/path` and `/indoor/factor_graph/segments`;
- `/indoor/odometry_only/path`;
- `/indoor/factor_graph/axes`;
- `/indoor/factor_graph/loops`;
- `/indoor/factor_graph/recovery_markers`.

The bag player publishes every recorded topic by default. Use an allow-list
when a large bag contains unnecessary sensors:

```bash
./pipeline.sh play \
  --bag /absolute/path/to/rosbag2 \
  --topics /odometry/filtered,/imu/data,/ouster/imu,/scan \
  --domain-id 42
```

The storage plugin is normally detected from `metadata.yaml`. If necessary,
add `--storage sqlite3` or `--storage mcap`.

## Set the pose in RViz

When the exact initial pose is not known, start the graph with:

```bash
./pipeline.sh run \
  --map /absolute/path/to/map.yaml \
  --output /absolute/path/to/output \
  --wait-for-initial-pose \
  --domain-id 42
```

Open RViz, select **2D Pose Estimate**, click the robot position on the map and
drag the arrow in its forward direction. Start the bag replay after setting
the pose.

## Live robot

For live ROS topics, omit the `play` command and disable simulated time:

```bash
./pipeline.sh run \
  --map /absolute/path/to/map.yaml \
  --output /absolute/path/to/output \
  --initial-pose X Y YAW_DEG \
  --domain-id 42 \
  --live

./pipeline.sh rviz --domain-id 42 --live
```

The container uses host networking. Configure the robot and host with the
same `ROS_DOMAIN_ID`, compatible DDS middleware and network visibility.

## Evaluate a completed run

After stopping localization with `Ctrl+C`:

```bash
./pipeline.sh evaluate \
  --output /absolute/path/to/output \
  --bag-label my_bag \
  --map-label my_map
```

The output directory contains:

```text
trajectory.csv
odometry_only.csv
status.json
summary.json
trajectory_preview.png
```

Acceptance rate and wall distance are consistency indicators, not absolute
accuracy. Proper validation requires independent ground truth or surveyed
checkpoints.

## Factor graph

Each state `Xi = (x, y, yaw)` is connected to the previous state by odometry.
A robust scan-to-map prior anchors accepted keyframes to the fixed occupancy
map. IMU yaw can be enabled when it is not already fused into odometry.
Verified loop closures add robust between-factors to old keyframes.

```text
Prior -> X0 -- odometry -- X1 -- odometry -- X2 ...
          |                |                |
       scan-map         scan-map         scan-map
          \____________ loop closure ______/
```

After consecutive scan failures, the validated recovery method performs a
global map search. It must produce three coherent high-quality hypotheses
before re-anchoring the graph. AMCL recovery remains disabled in the supplied
profile because it increased acceptance statistics without improving the
geometric trajectory in the hospital experiment.

## Adapting the profile

Start by changing only topic names. Tune noise and matching thresholds only
after checking timestamps, frames, the LiDAR extrinsic, map resolution and the
initial pose. Lower sigma means a stronger factor.

The supplied profile assumes a single approximately planar floor. It does not
estimate height, roll or pitch and is not suitable for stairs, ramps or
multi-floor 6D localization without extending the state model.
