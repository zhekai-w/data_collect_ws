# DualSense Teleop: Madgwick Filter, Controller Mapping, and MoveIt Servo

## Overview of the pipeline

This isn't one script — it's a chain of ROS 2 nodes (mostly in the sibling `dualsense_teleop` package) that turns raw controller motion into MoveIt Servo joint commands:

```
DualSense HW
 ├─(joystick)─────────────────────────────────► dualsense_pose_tracking.py ─┐
 └─(gyro+accel)→ dualsense_publish_imu.py                                  │
                    │ /imu/data_raw (Imu, no orientation)                  │
                    ▼                                                      │
              imu_filter_madgwick (external ROS pkg, Madgwick AHRS)        │
                    │ /imu/data (Imu, orientation filled in) + TF          │
                    ▼                                                      │
              imu_orientation_offset.py (calibration + L1 clutch + lock)   │
                    │ /imu/data_offset (Imu, orientation only)             │
                    └────────────────────────────────────────────────────►│
                                                                            ▼
                                                          /target_pose (PoseStamped)
                                                                            │
                                                          moveit_servo::PoseTracking
                                                          (pose_tracking_demo.cpp)
                                                                            │
                                                     Cartesian twist (PID, ur_servo/PID yaml)
                                                                            │
                                                          MoveIt Servo Jacobian IK
                                                                            │
                                              /forward_position_controller/commands
```

## 1. Madgwick filter

It isn't custom code in this repo — the launch file just instantiates the stock `imu_filter_madgwick` ROS package as a node (`dualsense_teleop/launch/dualsense_teleop.launch.py:31-42`):

```python
Node(package='imu_filter_madgwick', executable='imu_filter_madgwick_node', ...
     parameters=[{'use_mag': False, 'publish_tf': True, 'world_frame': 'enu', 'fixed_frame': 'world'}])
```

**I/O**

| | Topic | Type | Notes |
|---|---|---|---|
| in | `/imu/data_raw` | `sensor_msgs/Imu` | angular_velocity (rad/s) + linear_acceleration (m/s²) from `dualsense_publish_imu.py`; orientation field is left zeroed (`orientation_covariance[0] = -1`) since the DualSense has no absolute orientation |
| out | `/imu/data` | `sensor_msgs/Imu` | same gyro/accel passed through, but `orientation` now filled with the fused quaternion |
| out | TF | `world → imu frame` | because `publish_tf: true` |

Since `use_mag: False`, this runs Madgwick's **IMU-only** algorithm (gyro + accel), not the MARG (+magnetometer) variant — meaning yaw has no absolute reference and will drift, only roll/pitch are accelerometer-corrected.

**Math** (per IMU sample, quaternion `q = [q0,q1,q2,q3]` = world→sensor orientation):

1. Gyro integration rate:
   `q̇_ω = ½ · q ⊗ [0, ωx, ωy, ωz]`

2. Accelerometer error function (compares measured gravity direction `â` to gravity predicted by current `q`, reference `g=[0,0,1]`):
   ```
   f(q, â) = [ 2(q1q3 − q0q2) − âx ]
             [ 2(q0q1 + q2q3) − ây ]
             [ 2(½ − q1² − q2²) − âz ]
   ```
   with Jacobian
   ```
   J(q) = [ −2q2   2q3  −2q0   2q1 ]
          [  2q1   2q0   2q3   2q2 ]
          [   0   −4q1  −4q2    0  ]
   ```

3. Gradient-descent correction step:
   `∇f = Jᵀf`, normalized `∇̂f = ∇f / ‖∇f‖`

4. Fused derivative and integration:
   `q̇_est = q̇_ω − β·∇̂f`
   `q_{t+1} = normalize(q_t + q̇_est·Δt)`

`β` (the package's `gain` param, default ~0.1 here since unset) trades gyro-drift vs accelerometer-noise sensitivity — the accelerometer only ever corrects roll/pitch (it can't observe yaw), which is why this whole system needs the *next* stage to fix yaw drift.

## 2. IMU → orientation conditioning (`imu_orientation_offset.py`)

Subscribes `/imu/data`, publishes `/imu/data_offset` (`dualsense_teleop/dualsense_teleop/imu_orientation_offset.py:88-104`). Does three things:
- **Auto-calibration**: on the first message, looks up TF `world → dualsense_imu_frame` and stores it as `reference_rotation`, so whatever orientation you're holding the controller in at startup maps to "zero."
- **L1 clutch (mouse-lift behavior)**: while L1 is held, output = `held_rotation · (last_frame⁻¹ · current_frame)` — i.e., it tracks *relative* rotation deltas from the moment you press L1. Release L1 and the orientation freezes (`tracking_enabled=False`), so you can release the trigger, re-orient your wrist physically, and re-grab L1 to keep rotating from where you left off — exactly like lifting a mouse.
- **D-pad axis lock**: freezes two of {roll, pitch, yaw} at their locked-in value so you can rotate about only one axis at a time.

## 3. DualSense → EE translation/rotation (`dualsense_pose_tracking.py`)

- **Rotation**: taken wholesale from `/imu/data_offset` (line 335: `target_pose.pose.orientation = self.imu_orientation`) — i.e., tilting/twisting the physical controller directly sets the commanded wrist orientation.
- **Translation**: read directly from the joystick (not IMU), as a **velocity command integrated into a position target** (lines 308-332):
  - Left stick Y → ΔZ (up/down)
  - Right stick Y → ΔX (forward/back)
  - Right stick X → ΔY (left/right)
  - `delta = stick_axis × linear_scale × dt`, added each 30 Hz tick to the EE's *current* pose (fetched live via TF `base_link → tool0`).
- **R1**: snaps target to a fixed `home_pose` and locks orientation to home's orientation while held.
- Publishes `geometry_msgs/PoseStamped` to `/target_pose` at 30 Hz.

## 4. MoveIt Servo Pose Tracking

`replace_cpp/pose_tracking_demo.cpp` is a lightly modified version of MoveIt Servo's stock `pose_tracking_example.cpp` — it just waits for `/target_pose` externally instead of hardcoding a pose, then calls `tracker.moveToPose(lin_tol, rot_tol, timeout)` in a loop (line 175).

Internally, `moveit_servo::PoseTracking` (library code, not in this repo) runs independent PID loops per `config/pose_tracking_settings.yaml`:
- x/y/z position error → 3 PIDs, gain 20.0 P, 0 I/D, windup limit 0.05
- orientation error (angle of `current⁻¹ · target` as angle-axis) → 1 PID, gain 20.0 P

These PID outputs become a Cartesian `TwistStamped` velocity command fed into MoveIt Servo core, configured by `config/ur_servo.yaml`:
- `command_in_type: speed_units` — twist is already m/s / rad/s, not normalized joystick units
- Servo computes joint velocities via damped Jacobian pseudo-inverse IK, checks singularity (condition-number thresholds 100/200) and collisions (5 Hz, threshold-distance mode)
- Smooths with a Butterworth filter plugin
- Publishes joint **positions** (`publish_joint_positions: true`) as `std_msgs/Float64MultiArray` to `/forward_position_controller/commands` at 200 Hz (`publish_period: 0.005`)

So the division of labor is: DualSense sets *where* the EE should be/point at 30 Hz → PoseTracking's PID converts pose error into Cartesian velocity → MoveIt Servo's Jacobian IK converts that into safe joint commands at 200 Hz → `ros2_control` drives the UR5.
