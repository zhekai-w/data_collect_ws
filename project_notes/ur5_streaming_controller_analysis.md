# UR5 Streaming Control — Controller Analysis

Why the GR00T streaming client (`pkgs/Isaac-GR00T/scripts/ur5_gr00t_streaming_client.py`)
produced jerky action-by-action motion, what each UR ROS 2 controller actually does,
and why servoj streaming via `forward_position_controller` is the right interface for
VLA-style action streaming.

## The symptom

Policy actions arrive as chunks (e.g. 50 steps at `dt` = 70–160 ms per step) and are
dispatched one action per control tick. The arm visibly pulsed at the tick rate —
move, stop, move — regardless of:

- passing feedforward velocities on each trajectory point,
- sending a multi-point lookahead window (receding horizon) per tick,
- switching from the `FollowJointTrajectory` action interface to the JTC command topic,
- slowing the tick from `dt=0.07` to `dt=0.16`.

## What the data showed

Diagnostics were added to the client (per-tick commanded/measured positions, per-section
loop timing, high-rate `/joint_states` recording; see `scripts/plot_streaming_log.py`):

1. **Commanded trajectory is smooth** — ~0.01 rad per-tick deltas, no staircase.
   The policy + RTS chunk smoothing is not the problem.
2. **Client loop timing is healthy** — loop body < 2 ms per tick, ticks land on target
   (sleep overshoot ~8 ms). The client is not the problem.
3. **Joint velocity is a sawtooth locked to the goal rate** — velocity pulses to ~2×
   the average right after every goal send, then decays to ~zero before the next goal.
   The robot genuinely stops and restarts at every action.

Conclusion: the jerk is created inside the trajectory controller's handling of
*trajectory replacement*, not by the commands.

## UR ROS 2 controller landscape

| Controller | Interface | Mechanism | Fit for per-action streaming |
|---|---|---|---|
| `scaled_joint_trajectory_controller` (default) | `FollowJointTrajectory` action | Spline interpolation on the ROS PC, 500 Hz sampling, respects speed slider | ✗ — new goal preempts the old one: abort → hold → restart from ~zero velocity. Stop-and-go at the goal rate. |
| same, command topic (`~/joint_trajectory`) | `JointTrajectory` topic | Trajectory *replacement* without goal handshake | ✗ — empirically still resets velocity on each replacement at streaming rates (same sawtooth). Designed for occasional re-planning, not 6–14 Hz replacement. |
| `passthrough_trajectory_controller` | `FollowJointTrajectory` action | Forwards the whole trajectory to the robot controller; interpolation runs on the robot | ✗ for streaming (same replacement problem), ✓ for chunk-at-a-time replay: excellent spline execution with no RT load on the PC. |
| `forward_position_controller` | `Float64MultiArray` topic | Each setpoint forwarded to URScript `servoj()` in the driver's 500 Hz loop; `servoj` applies `lookahead_time`/`gain` smoothing | ✓ — the documented servoing interface (what MoveIt Servo uses). Requires a *dense* setpoint stream (100+ Hz), so sparse policy actions must be upsampled client-side. |
| `forward_velocity_controller` | `Float64MultiArray` topic | URScript `speedj()` | ✓ alternative; often smoother still, but needs a velocity-domain controller (P-loop on position error) and careful stop handling. |

Key facts from the UR driver docs:

- The forward controllers are "particularly useful when doing servoing such as
  moveit_servo" — they are the intended streaming interface.
- Streaming controllers **do not respect speed scaling** (teach-pendant slider) and do
  no feasibility checking: the sender is responsible for safe, achievable commands.
- `servoj` smoothness is tuned by hardware-interface launch args
  `servoj_lookahead_time` (0.03–0.2 s, higher = smoother + more lag) and `servoj_gain`.

## Why sparse trajectory goals can never be smooth here

A `FollowJointTrajectory` goal is a *contract to execute a full trajectory*. Sending a
new one per action forces the controller through its preemption path every tick:
abort the active goal, (briefly) hold, restart the new spline from the current
(measured) state with zero initial velocity. Feedforward velocities and lookahead
points inside the goal cannot help, because the discontinuity is injected by the goal
lifecycle itself, before the spline is ever sampled. The topic interface avoids the
action handshake but showed the same velocity reset per replacement on this driver
version.

## The fix implemented

`ur5_gr00t_streaming_client.py` now defaults to `--control_interface forward_position`
(the MoveIt Servo pattern):

- The control loop still pops one action per `dt`, but instead of sending a trajectory
  goal it publishes a *segment* (previous target → new target over `dt`).
- A dedicated streamer thread interpolates along the segment and publishes setpoints to
  `/forward_position_controller/commands` at `--stream_hz` (default 125 Hz).
- `servoj` lookahead smoothing inside the driver blends the remaining discretization.
- Homing also streams through the fpc (smoothstep profile), since the JTC is inactive.
- The JTC-topic path is retained under `--control_interface jtc_topic` for comparison.

### Usage

```bash
# one-time: activate the streaming controller
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate forward_position_controller

python scripts/ur5_gr00t_streaming_client.py \
  --lang "place apple in the basket." \
  --aggregate_fn_name weighted_average --chunk-filter rts \
  --host <server> --port 5555 \
  --action_horizon 50 --chunk_size_threshold 0.4 --dt 0.07
```

Safety notes: the speed slider does not limit fpc motion — keep the e-stop reachable on
first runs. If motion is smooth but slightly wobbly, raise the driver launch arg
`servoj_lookahead_time` (e.g. `0.1`).

### Diagnostics

Every run writes `streaming_log.npz` (commanded vs measured positions, loop-section
timings, goal/gripper timestamps, high-rate joint states). Analyze with:

```bash
python scripts/plot_streaming_log.py streaming_log.npz
```

`streaming_log_vel.png` shows per-joint velocity with command events overlaid — the
stop-and-go sawtooth is directly visible there if any interface change regresses.

## References

- [UR ROS 2 driver — controllers](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/usage/controllers.html)
- [UR ROS 2 driver — position/velocity control (servoj/speedj)](https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/rolling/doc/ur_robot_driver/ur_robot_driver/doc/usage/position_velocity_control.html)
- [MoveIt Servo tutorial](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html)
- [joint_trajectory_controller docs](https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)
- [UR driver information flow (ROS → robot)](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/discussions/489)
- [MoveIt Servo on UR5e smoothness issue](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/issues/912)
