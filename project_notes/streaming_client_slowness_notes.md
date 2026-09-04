# `ur5_gr00t_streaming_client.py` — slowness investigation

## Fixed: homing took ~10s instead of ~1s

File: `pkgs/Isaac-GR00T/scripts/ur5_gr00t_streaming_client.py`, `home_via_fpc()`.

**Root cause (two stacked issues):**
1. `duration` default was `4.0` — 4x slower than intended even in the best case.
2. The per-iteration `time.sleep(1.0 / hz)` did not compensate for loop overhead
   (unlike `_fpc_streamer_loop`, which measures elapsed time and subtracts it
   from the sleep). Under GIL contention from the `rclpy.spin` thread and camera
   threads, per-iteration overhead stacked on top of the sleep instead of being
   absorbed by it, stretching the nominal move time well past 4s.

**Fix applied:**
```python
def home_via_fpc(sensor: UR5SensorNode, duration: float = 1.0, hz: float = 125.0):
    """Move to home by streaming a smoothstep interpolation to forward_position_controller."""
    state = sensor.get_joint_state()
    cur = state[:6].astype(float)
    target = np.asarray(HOME_JOINT_POSITIONS, dtype=float)
    n = max(1, int(duration * hz))
    period = 1.0 / hz
    for i in range(1, n + 1):
        tick = time.perf_counter()
        a = i / n
        s = 3 * a * a - 2 * a * a * a  # smoothstep: zero velocity at both ends
        sensor.publish_fpc(cur + s * (target - cur))
        elapsed = time.perf_counter() - tick
        time.sleep(max(0.0, period - elapsed))
```
- `duration` default `4.0 → 1.0`.
- Elapsed-time-compensated sleep, same pattern as `_fpc_streamer_loop`.
- Both callers (`main()` startup home, `go_home()` on `'h'` keypress) use the
  default, so no other call sites needed changes.

**Status:** applied, not yet hardware-tested (not near the robot). Verify actual
wall-clock time on next session — should land close to 1s now.

---

## Open question: why does streaming_client "feel" slower than simple_client overall?

Filter choice (`chunk_filter=rts`) ruled out — user runs both clients with the
same filter, so that's not the differentiator.

Two architectural differences remain as suspects:

### 1. Interpolation location (GIL contention)
- `simple_client.py` (default `controller=scaled_joint_trajectory_controller`)
  sends a full trajectory as one `FollowJointTrajectory` action goal
  (`send_chunk_action`). Interpolation between waypoints happens **inside the
  ROS controller_manager's real-time C++ loop** — nothing in the Python
  process affects smoothness after the publish.
- `streaming_client.py` (default `control_interface=forward_position`)
  interpolates **in a Python thread** (`_fpc_streamer_loop`), publishing
  setpoints at `stream_hz=125` every 8ms. This competes for the GIL with the
  `rclpy.spin` thread and the two camera-decode threads (`_k4a_loop`,
  `_wfov_loop`), unlike simple_client's approach.
- streaming_client also runs more total threads (~7 vs ~5), increasing
  scheduling/GIL contention generally.

**Test to isolate:** run streaming_client with `--control-interface jtc_topic`
(same JTC *controller* as simple_client, but driven via the **topic** interface,
not simple_client's action interface — see "Three distinct concerns" below) and
compare. If motion feels smoother, GIL contention from the Python-side 125Hz
streamer is confirmed as the cause.

### 2. `aggregate_fn_name="conservative"` blending
- `streaming_client` (line ~73) defaults to blending overlapping timesteps
  `0.7 * old + 0.3 * new` when a new inference chunk lands mid-execution.
  This biases toward the older (potentially stale) prediction on every
  overlap.
- `simple_client` has no equivalent — it always executes one full chunk
  before requesting the next, so there's never an overlap to blend.
- Independent of `chunk_filter`, this adds lag whenever inference chunks
  overlap the pending queue (i.e. whenever `chunk_size_threshold` triggers
  early re-inference).

**Test to isolate:** run streaming_client with `--aggregate-fn-name
latest_only` and compare responsiveness to a course correction.

### Next steps (once back at hardware)
- [ ] Confirm `home_via_fpc` now takes ~1s.
- [x] A/B test `--control-interface forward_position` vs `jtc_topic` — **done,
      forward_position wins**; see "Hardware result" below.
- [ ] A/B test `--aggregate-fn-name conservative` vs `latest_only`.

---

## Control interface: forward_position vs jtc_topic

### Controller loop rate is 125 Hz, NOT 200 Hz
- `ur5_update_rate.yaml`: `controller_manager: update_rate: 125`.
- The 200 Hz figure elsewhere is moveit_servo's *publish* rate, not the
  ros2_control loop. On real UR, robot-side RTDE is 500 Hz, but the ROS
  controller_manager driving it here is 125 Hz.

### Why simple_client default "looks like an approximation"
simple_client default = `send_mode=chunk` + `scaled_joint_trajectory_controller`
via the **action** interface (`send_chunk_action`, blocking). Two artifacts:
1. **Spline rounding** — JTC fits a cubic/quintic spline through waypoints and
   invents its own velocities if none given → rounds VLA corners.
2. **Chunk-boundary pause + jump** — sends whole 16-pt chunk as one blocking
   goal, waits, *then* infers next chunk → robot sits at chunk end during
   inference, plus discontinuity at boundary.

### Why forward_position looks authentic
`forward_position_controller` (JointGroupPositionController) does NO
interpolation/spline — forwards raw published positions. `_fpc_streamer_loop`
lerps between exact VLA waypoints at 125 Hz → path passes exactly through VLA
points. Async inference overlaps execution → no boundary pause. Cost:
(a) velocity discontinuity (jerk) at every waypoint; (b) GIL jitter from the
Python 125 Hz interpolator.

### jtc_topic = the middle ground
`--control-interface jtc_topic` = JTC controller + **topic** interface
(`send_window_scaled_joint` → `/scaled_joint_trajectory_controller/joint_trajectory`)
+ client-supplied finite-difference velocities. Gains:
- C++ 125 Hz interpolation (no GIL jitter — moves interp off Python)
- no chunk pause: topic trajectories **replace** the active one seamlessly
  (re-sampled from current desired state), unlike action preemption which
  inserts hold + zero-vel restart (stop-and-go)
- spline *through* your points *at your commanded velocities* → near-authentic
  (corners honored), without raw-FPC velocity jerk

Tradeoff vs FPC: JTC still splines, so not bit-exact to VLA points — but much
closer than simple_client default because you pin the velocities.

### Three distinct concerns (do not conflate)
1. **Which controller active** (robot side): JTC vs FPC — spawn/activate one.
2. **Client interface** (client side): `--control-interface` arg.
3. **How JTC is driven**: action interface (blocking goal, simple_client) vs
   topic interface (streaming replace, jtc_topic). Same controller, different
   feel. Topic interface is the one that streams smoothly.

## How to enable jtc_topic — NO code change needed (already implemented)
1. Client: `python ur5_gr00t_streaming_client.py --control-interface jtc_topic`
2. Robot: activate `scaled_joint_trajectory_controller`, deactivate fpc:
   ```bash
   ros2 control switch_controllers \
     --activate scaled_joint_trajectory_controller \
     --deactivate forward_position_controller
   ```
   (dualsense launch spawns ONLY fpc — lines 289/297; scaled is commented out.
   Whatever launch brings up the robot for streaming_client must activate JTC.)
- `_control_loop` already branches on `control_interface` (line ~421): fpc sets
  a segment for the Python streamer; else builds finite-diff velocities and
  calls `send_window_scaled_joint`.
- fpc streamer thread only starts in fpc mode (`run`, line ~485) — correctly
  skipped for jtc_topic.
- Homing already branches (`go_home` ~356 / `main` ~583): jtc_topic homes via
  the action interface (`send_single_action_scaled_joint`), needs JTC active.

## Blending: does the control loop know which action is executing?
Knows exactly which action was **DISPATCHED** (popped), atomically under lock.
Does NOT read robot physical state — "dispatched" is assumed = "executing."
Identical bookkeeping in both fpc and jtc_topic modes; only physical
realization differs (JTC spline vs Python lerp).

**Gap 1 — inference latency (handled).** New chunk built on the receiver
thread; `base_step` snapshotted at inference *start* (line ~251), before the
blocking ZMQ call. Control loop keeps popping during inference →
`latest_executed_step` advances past `base_step`. The merge
(`_aggregate_into_queue`, ~189) re-reads `latest_executed_step` FRESH under
lock and drops any incoming action `<= latest` (~199). So at merge instant it
knows the exact last-dispatched action and blends only strictly-future pending
points. Leading N ≈ inference_latency/dt points of each chunk are discarded.
This is the docstring's "no wall-clock estimation" — pop counter IS the exact
dispatch truth.

**Gap 2 — physical tracking lag (unmodeled, ignored).** Robot physically lags
the dispatched command (controller tracking, servo lag). Never measured from
`/joint_states`. Irrelevant for blending because you blend *future* points the
arm hasn't reached; lag on already-dispatched points doesn't touch future
timesteps.

**Real limiter:** if inference latency is large vs queue drain, you discard a
big chunk head each cycle → effective fresh horizon shrinks to
`(action_horizon − inference_latency/dt)`. If that gets small, motion leans on
stale predictions. This — not controller choice — is the thing to watch.

### jtc_topic tuning notes
- If JTC spline rounds too much vs raw VLA: drop `--lookahead` toward 2, or
  shorten `--dt`.
- Verify JTC seamless-replace looks smooth at `dt=0.05`, `lookahead=4` on hw.

---

## Hardware result: jtc_topic is WORSE than forward_position, not better

Tested on real hardware:
```
python scripts/ur5_gr00t_streaming_client.py \
  --lang "place apple in the basket." \
  --aggregate_fn_name weighted_average \
  --chunk-filter rts \
  --host 163.13.164.145 --port 5555 \
  --action_horizon 50 --chunk_size_threshold 0.6 \
  --dt 0.07 --filter --filter_mincutoff 0.5 \
  --control_interface jtc_topic
```
Result: visible jitter, motion does not smooth out — worse than
`--control_interface forward_position` (the default). This contradicts the
"GIL contention" hypothesis above (the one motivating the "test to isolate"
in the Interpolation location section) — jtc_topic does NOT look smoother
despite moving interpolation out of Python and into the C++ controller loop.

### Root cause: `open_loop_control` defaults to false → closed-loop resampling

`send_window_scaled_joint` (simple_client.py, reused by streaming_client) is
called every control-loop tick (~14 Hz at `dt=0.07`) in jtc_topic mode,
republishing a fresh multi-point `JointTrajectory` on the JTC command topic —
each call fully replaces the currently-executing trajectory.

Repo controller config — `src/dualsense_teleop/config/ur5_real_controllers.yaml:109-135`
and `ur5_ros2_controllers.yaml:40-66` — never sets `open_loop_control` (both
are byte-identical copies of the upstream `ur_robot_driver` default, confirmed
against `/opt/ros/humble/share/ur_robot_driver/config/ur_controllers.yaml`).
Unset → ros2_control default applies: `open_loop_control = false`
(`/opt/ros/humble/include/joint_trajectory_controller/joint_trajectory_controller_parameters.hpp:79`),
i.e. **closed-loop**.

Closed-loop means every replan (every ~70 ms) re-anchors the new cubic spline's
start position/velocity boundary condition from the *actually measured* joint
state (`state_interfaces: [position, velocity]`,
`ur5_real_controllers.yaml:120-122`) rather than from the controller's own
previously-commanded desired state. Measured velocity is noisy
(differentiated encoder / RTDE feedback), so that noise gets re-injected into
the trajectory's start-velocity boundary condition on every single replan —
visible as jitter. `forward_position` mode never has this problem:
`_fpc_streamer_loop` (streaming_client.py:283-298) always continues from its
own Python-tracked `self._last_target`, fully open-loop relative to sensor
feedback.

### Tried the fix — `open_loop_control: true` — it crashes the driver

Added `open_loop_control: true` under `scaled_joint_trajectory_controller.ros__parameters`
in `ur5_real_controllers.yaml`. On launch, `ur_ros2_control_node` aborted
immediately on controller activation (before any trajectory was sent):

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  can't compare times with different time sources
...
in joint_trajectory_controller::Trajectory::sample(...)
in ur_controllers::ScaledJointTrajectoryController::update(...)
```

Installed versions: `ros-humble-ur-controllers 2.13.0`,
`ros-humble-joint-trajectory-controller 2.53.1` — already fairly current, so
not a simple "stale package" problem. A Universal Robots forum thread reports
the identical stack/error ("ROS2 driver issue: can't compare times with
different time sources"), with a fix that required rebuilding several UR
packages from source with matched versions — a bigger undertaking than a
config flip. Likely cause: `ScaledJointTrajectoryController::update()` (UR's
speed-scaling override) samples using a node-clock (`RCL_ROS_TIME`) time,
while JTC's open-loop last-commanded-state path seeds/samples using a
differently-typed `rclcpp::Time` (e.g. default-constructed `RCL_SYSTEM_TIME`)
— `rclcpp::Time` throws on comparison whenever clock *types* differ, even if
the underlying wall-clock values would agree.

**Decision: `open_loop_control` left commented out in both yaml files.** Not
safely usable with this driver stack right now — do not re-enable without a
from-source rebuild investigation.

### Path forward for jtc_topic jitter (without open_loop_control)
- [ ] Try republishing the jtc_topic window every 2-3 ticks instead of every
      `dt` — reduces how often the controller re-anchors to noisy measured
      state; the multi-point window already gives it a few ticks worth of
      future waypoints to execute smoothly in between replans.
- [ ] Otherwise, prefer `forward_position` for now — it's the smoother option
      on hardware as tested, jtc_topic's closed-loop resampling issue can't be
      fixed cleanly without `open_loop_control`.

---

## forward_position jerk: the linear lerp between waypoints

With jtc_topic ruled out, the remaining roughness on `forward_position` is
cost (a) from the "Why forward_position looks authentic" section above —
*"velocity discontinuity (jerk) at every waypoint"* — which had been listed but
never acted on.

`_fpc_streamer_loop` published `frm + alpha*(to - frm)`: a linear lerp, so
velocity is piecewise **constant** and steps discontinuously at every waypoint,
i.e. every `dt` = 50-70 ms. Everything upstream (`rts_smoother_chunk` on the
inference chunk, the optional `OneEuroFilter` at `freq = 1/dt` = 14-20 Hz) smooths
the *waypoints*, and then the lerp puts the corners back.

### Why the DualSense teleop stack never has this

Same endpoint (`/forward_position_controller/commands` → `servoj`), different
place in the pipeline for the smoothing:

| | teleop | streaming client (before) |
|---|---|---|
| setpoint source | `measured_joints + J⁺·twist·Δt`, closed loop @200 Hz | policy waypoint, lerped @125 Hz |
| where it filters | **Butterworth on the dense 200 Hz output stream** (`config/ur_servo.yaml:35,44` — `ButterworthFilterPlugin`, `low_pass_filter_coeff: 10.0`) | at the sparse 14-20 Hz waypoint rate |
| after the filter | nothing — publish | **the lerp**, which re-adds corners |
| robot side | `servoj(gain, lookahead_time)` | same |

Teleop's error is also inherently bounded: `dualsense_pose_tracking.py` integrates
`/target_pose` off a *live* TF `base_link→tool0` each 30 Hz tick, so full stick
deflection asks for only ~0.1 m ahead of the actual arm, and
`pose_tracking_settings.yaml` turns that into twist with a P-only gain of 20.
The single portable idea is the middle row: **filter after the setpoint generator,
on the dense stream.**

### What was implemented

`ur5_gr00t_streaming_client.py` + `filter_utils.py`:

1. **C1 cubic Hermite segments** replace the lerp. Segment tuple is now
   `(t0, dur, p0, v0, p1, v1)`; the outgoing tangent `v1` is a Catmull-Rom
   finite difference over the `lookahead` window the control loop already peeks
   (`window[1]`), with Fritsch-Carlson monotone limiting. The next segment
   inherits `(p0, v0)` by *sampling the live segment* rather than from stored
   bookkeeping, so continuity is exact even when a tick is late. Only the
   outgoing tangent is limited — clipping the inherited `v0` would re-break the
   continuity this exists to provide.
2. **`ButterworthLPF` at `stream_hz`** — 2nd-order, per joint, applied to every
   published setpoint right before `publish_fpc`. This is the direct analogue of
   moveit_servo's plugin. Seeded with `scipy.signal.lfilter_zi` on every reset;
   a zero-state reset would ramp from 0 rad toward the current joint angle, i.e.
   command a swing through the origin.
3. **Brake tail on overrun.** A cubic evaluated past `s=1` runs away, but a hard
   clamp snaps velocity to zero whenever a tick is late. Instead a
   constant-deceleration tail runs for `brake_time`, C1 at the junction and
   settling to a fixed point — so the pause path still holds position by
   republishing.
4. **Reseeds on measured state** at startup, after `go_home`, and on resume from
   pause (the arm may have been freedriven while paused). These are one-shot,
   position-only, taken while the arm is stationary.
5. `--filter` (the sparse OneEuro) is auto-disabled with a warning when a stream
   filter is active: it rewrites `window[0]` but not `window[1]`, so the Hermite
   tangent would mix filtered and unfiltered points.

**Still fully open-loop** between reseeds. Re-anchoring `p0`/`v0` on measured
`/joint_states` every tick is exactly the jtc_topic failure mode diagnosed above:
servo gets away with it because it re-anchors at 200 Hz *and* low-passes the
incoming state first; doing it at 14-20 Hz behaves like an impulse train instead.
The real outer loop already exists — the VLA conditions each chunk on measured
state at inference time.

### A/B flags (both paths live, no rebuild needed)

```bash
# before — reproduces the old lerp bit-for-bit
--segment-interp linear  --stream-filter none
# interpolation only
--segment-interp hermite --stream-filter none
# after (new defaults)
--segment-interp hermite --stream-filter butter --stream-filter-cutoff 8
# filter only
--segment-interp linear  --stream-filter butter --stream-filter-cutoff 8
```

`streaming_log.npz` now also carries `stream_t` / `stream_cmd` (the dense
published stream, gated by `js_recording`) plus a config stamp
(`segment_interp`, `stream_filter`, `stream_filter_cutoff`, `dt`, `stream_hz`), so
a saved log is self-identifying. `plot_streaming_log.py` prints the metrics and
writes `*_stream.png`. The existing `cmd` array is the *waypoint* stream and is
blind to all of this — it should be unchanged across the A/B.

One metric caveat worth keeping: RMS jerk alone is a poor discriminator between a
lerp and a spline. The lerp is jerk-free *inside* each segment and fires an
impulse at every waypoint; a cubic has a modest jerk everywhere; squaring and
pooling flattens the difference. `plot_streaming_log.py` reports `max|jerk|` and
`max|Δv|` alongside it for that reason.

### Hardware result: hermite + butter did NOT fix it

Tested `--segment-interp hermite --stream-filter butter` on hardware: **buzzing
and slowness both persist**, less pronounced than `linear + none` but the same
problem. Assume the remaining combinations behave the same — no point running the
full A/B matrix.

**Conclusion: waypoint-boundary jerk was not the dominant cause.** Smoothing the
*value* sequence helps a little and stops there.

### Leading hypothesis: publish TIMING, not setpoint values

`ButterworthLPF` smooths the sequence of values; it has no influence on *when*
they reach the controller. Jitter introduced after the filter — scheduling,
transport, controller read cadence — passes through untouched. Two mechanisms,
both invisible in `stream_cmd`:

1. **`stream_hz = 125` into `update_rate = 125`.** Two free-running clocks at the
   same nominal rate drift in phase, so setpoints consumed per control cycle go
   `1,1,2,0,1,0,2,...` instead of a steady `1,1,1`. A starved cycle re-commands
   the stale value (zero velocity for 8 ms); a doubled cycle jumps two steps. FPC
   is a pure passthrough and cannot bridge the gap.

   *Corrects an earlier claim in this file* that `stream_hz > 125` is "wasted,
   and above 125 you alias". moveit_servo publishes at **200 Hz into this same
   125 Hz loop** (`ur_servo.yaml:22`). Oversampling is deliberate — it guarantees
   every cycle has a fresh value. Matched rates are the bad case.

2. **GIL jitter.** `_fpc_streamer_loop` targets 8 ms while competing with
   `rclpy.spin` and the two camera-decode threads. `_sample_segment` uses
   wall-clock `t`, so each value is correct for its instant — but the controller
   holds it until the next arrives, and non-uniform hold durations of a correct
   curve still give non-uniform velocity.

Scale: injecting a 4% rate of 4-20 ms stalls into an otherwise ideal stream moves
RMS jerk from 1.1 to 5560 with byte-identical commanded values. Timing dominates
value-domain smoothing by orders of magnitude.

### Diagnostic — the existing log already answers this

`stream_t` is recorded per published setpoint; `plot_streaming_log.py` now prints
the interval distribution and flags both jitter and the matched-rate case.

- median ≈ 8 ms, p99 < 10 ms → timing is fine, look elsewhere
- p99 > 12 ms or a fat tail → mechanism 2
- clean 8 ms but buzz persists → mechanism 1

### RESOLVED: jtc_topic jitter was `open_loop_control` after all

The `open_loop_control: true` crash was a **one-line bug in ur_controllers**, not a
config or version problem. `ScaledJointTrajectoryController::on_activate` does
`last_commanded_time_ = rclcpp::Time()` — default-constructed, so `RCL_SYSTEM_TIME`.
The open-loop branch of `update()` feeds that into
`set_point_before_trajectory_msg()`, and the following `sample(traj_time_)` compares
it against `RCL_ROS_TIME` — `rclcpp::Time` throws whenever clock *types* differ. The
closed-loop branch passes `time` directly, so it never mismatches. JTC installs a
hold trajectory at activation, hence the crash before any trajectory is sent.

Fix: `last_commanded_time_ = get_node()->now();` — patched copy of the driver at tag
2.13.0 in `src/Universal_Robots_ROS2_Driver` (only `ur_controllers` builds; the rest
carry `COLCON_IGNORE` and stay on the 2.13.0 binaries). No upstream fix exists;
checked 2.13.0..2.13.2 and the issue tracker.

**Hardware: with the patch + `open_loop_control: true`, jtc_topic jitter is gone.**
This confirms the closed-loop-resampling diagnosis. Remaining issue is that the
trajectory itself is not very smooth — a different problem, see below.

### Suspect for remaining roughness: constant-weight chunk blending

`_aggregate_into_queue` applied the SAME old/new weight at every overlapping
timestep, which steps the queue at both ends of the overlap on every merge. With
old flat at 0.0 and a new chunk disagreeing by 0.10 rad over a 20-step overlap:

| aggregate | first pending waypoint | max step between waypoints |
|---|---|---|
| `weighted_average` | 0.070 | 0.030 |
| `conservative` | 0.030 | 0.070 |
| `latest_only` | 0.100 | 0.000 |
| `ramp` (new) | 0.0048 | 0.0048 |

`weighted_average` jumps 0.07 rad at the very next timestep — ~1 rad/s of commanded
velocity spike at `dt=0.07`, once per merge. Chunk filtering runs *before* this, so
nothing smooths it. Added `--aggregate-fn-name ramp`: sweeps the weight 0->1 across
the overlap, so it is continuous with what is executing and converges to the new
chunk by the pure-new region.

Also added: merge events logged to `streaming_log.npz` as `merge` (time,
latest_step, n_blended, n_new, n_dropped); `plot_streaming_log.py` draws them as
purple dashed lines on the velocity panel and prints waypoint-vs-measured smoothness
separately.

### Comparing against simple_client chunk mode — `scripts/compare_logs.py`

With `open_loop_control: true`, jtc_topic is smoother but still rougher than
`ur5_gr00t_simple_client.py --send-mode chunk`. Both now use the same controller,
the same `--chunk-filter rts` and the same `--dt 0.07`, so the difference is purely
in dispatch: one 30-point goal per chunk vs a 4-point window republished every tick
(~14 Hz), `boundary_blend` once per chunk vs `weighted_average` on every merge, and
simple_client **stops and dwells** between chunks (blocking goal + `time.sleep(1.0)`)
while the streaming client never stops.

`compare_logs.py A.npz B.npz` scores both. Three corrections it makes, without which
the comparison is meaningless:

- **Dwell masking.** simple_client's stationary stretches would deflate every pooled
  RMS. Metrics run on moving segments only, trimmed 100 ms at each end so the per-chunk
  start/stop ramps are not scored as roughness.
- **Reconstruction.** simple_client logs the *raw* chunk; the streaming client logs
  post-rts, post-blend waypoints. `compare_logs.py` replays `rts_smoother_chunk` +
  `blend_chunk_boundary` per chunk so both `cmd` series mean the same thing. (Measured
  on synthetic data: per-chunk rts alone raises commanded rms jerk 1.1 → 86 through
  smoother edge transients at every chunk boundary — this affects *both* clients.)
- **Lag removal.** The tracking residual cross-correlates first, so it reports tracking
  error rather than the fixed transport delay.

The discriminator is the **velocity PSD**, not the jerk score: a peak at 14.3 Hz
(= `1/dt` = the republish rate) means the per-tick republish, a peak at the merge rate
means blending, broadband + a bad tick-interval p95 means Python loop timing. Verified
on synthetic logs: a 1.5 mrad 14 Hz ripple shows up as the top PSD peak at 14.16 Hz and
rms jerk 1.1 → 11190, while a pure-hold log yields zero moving samples instead of a
flattering near-zero score.

Runs to do, each walking the streaming client one variable closer to simple_client:
`--lookahead 30` (dispatch horizon), then `--chunk_size_threshold 0.15` (merge rate),
then `--aggregate_fn_name ramp` / `latest_only`, then `--action_horizon 30`. Whichever
step closes the gap is the answer. Note simple_client needs `--log` or it writes no npz.

### MEASURED: the loop was running 31% slow, and it was the GIL

First real A/B (`compare_logs.py simple_client_log.npz streaming_log.npz`):

```
per-tick section (ms):  pop 0.01   send 0.23   grip 0.00   log 0.04
                        sleep_req 69.66   sleep_over 22.00 (p95 30.2)
tick interval: median 92.1 ms (nominal 70.0)  ->  10.86 Hz, requested 14.29
```

0.3 ms of work per tick. The other 22 ms is `time.sleep()` overshoot — the sleep
returns only when the thread can reacquire the GIL, and the spin thread is doing 500
Hz `/joint_states` plus camera frames in Python. Reproduced: a 70 ms sleep overshoots
0.07 ms in an idle process and 2.4 ms median / 17 ms p95 with two GIL-busy threads.

Corroborating evidence in the same logs: `/joint_states` was recorded at **35–38 Hz**,
not the driver's 500. ~93% of messages dropped. That also makes every measured-side
number in those two logs unusable — Nyquist 18 Hz, so the 14.3 Hz republish artifact
is at the fold and the PSD peaks near 15–19 Hz are aliases.

**This is why simple_client is smoother, and it is not about interpolation.**
simple_client hands the controller a 30-point goal and the controller clocks the
motion in C++ at 125 Hz; Python timing only decides when the next *chunk* starts. The
streaming client made Python responsible for every waypoint's deadline.

Commanded-side (rate-independent, so still valid): B rms jerk 8.2 / max 92.3 vs A 2.1
/ 22.1. B's *typical* jerk is lower (rts on 50-point chunks) — the gap is spikes, and
they cluster: p90 jerk 33.9 within 250 ms of a merge vs 15.0 elsewhere. So blending
is real but second-order next to the timing.

### Fix

1. **Absolute tick deadline** instead of `sleep(dt - elapsed)`. The old form makes
   every tick *at least* dt, so overshoot is permanent playback stretch; a deadline
   lets the next tick absorb it. This alone restores the rate.
2. **Busy-wait tail** (`--tick-slack`, default 5 ms): sleep to `deadline - slack`,
   then spin. Makes the wake independent of GIL reacquisition.
3. **`sys.setswitchinterval(0.001)`** (`--gil-switch-interval`) — the default 0.005
   is the same order as the error being removed.
4. **Cheap `_jointstate_callback`**: resolve the name→index mapping once instead of
   rebuilding a dict and a list comprehension 500x/s. Cuts the GIL load at the source
   and should restore the js logging rate that the diagnosis depends on.

Measured under three GIL-hog threads, `dt=0.07`:

| | median err | p95 | effective rate |
|---|---|---|---|
| old: `sleep(dt)` | 6.14 ms | 20.3 ms | 12.79 Hz |
| deadline only | 6.12 ms | 25.3 ms | **14.27 Hz** |
| + 5 ms busy-wait | 1.79 ms | 23.2 ms | 14.23 Hz |
| + switchinterval 0.001 | **0.97 ms** | 13.6 ms | 14.29 Hz |

The deadline fixes the *rate* (the slowness); the busy-wait and switch interval fix
the per-tick *jitter* (a roughness source). `prof` column 6 is now the signed deadline
error, not the sleep overshoot. `--tick-slack 0` reproduces the old wake behavior.

Not done: C++. The real-time work already happens in C++ inside
`scaled_joint_trajectory_controller` at 125 Hz — Python only has to hand it trajectory
in advance. A rewrite would be warranted only for the `forward_position` path, where
Python is genuinely on the critical path every 8 ms.

### Result of the timing fix, and what it did not fix

`--lookahead 30 --chunk_size_threshold 0.4`, timing fix in:

```
tick ivl mean 70.04 ms -> 14.28 Hz (target 14.29)     was 10.86 Hz
tick_err median 8.20  p95 21.10  max 28.59
```

**Rate fixed.** The absolute deadline did it — playback is no longer stretched.

**Jitter not fixed.** The 5 ms busy-wait tail never fires: `time.sleep` returns
already past the deadline. Raising `--tick-slack` to 25 ms would work mechanically but
would spin holding the GIL for 36% of every tick, starving the spin thread further.
Wrong lever. `send`/`grip`/`log` all show 23–28 ms maxima *together*, so it is
whole-process stalls (GC or GIL), not one slow section. `/joint_states` is still 35 Hz
and bimodal — 311 gaps < 5 ms against 463 gaps of 25–35 ms — i.e. the subscriber is
blocked ~30 ms then drains a backlog. Same root cause; the callback was never the
bottleneck, the thread simply is not scheduled.

### Absolute waypoint schedule (`--jtc-absolute-timing`, default on)

Rather than keep fighting the jitter, stop letting it matter. `_window_times` puts
each waypoint on a fixed wall clock — waypoint `ts` is due at
`anchor_wall + (ts - anchor_step) * dt` — and publishes `time_from_start = due - now`.

| tick | old: `(i+1)*dt` | new, 20 ms late each tick |
|---|---|---|
| 0 | 0.07 0.14 0.21 0.28 | 0.07 0.14 0.21 0.28 |
| 1 | 0.07 0.14 0.21 0.28 | 0.05 0.12 0.19 0.26 |
| 2 | 0.07 0.14 0.21 0.28 | 0.03 0.10 0.17 0.24 |
| 4 | 0.07 0.14 0.21 0.28 | drop 1 waypoint, resync to 0.06 |

Late publish shortens the first segment instead of pushing every future point, so the
error stops accumulating. Already-due points are **dropped, not clamped** — clamping
would re-stretch the timeline, which is the thing this exists to remove. Keep the
clock, drop frames. A stall longer than the whole window re-anchors instead of firing
a catch-up sprint. On-time ticks reproduce the old numbers exactly, so this is a no-op
when timing is good. Drop count is printed at exit and `--no-jtc-absolute-timing`
restores the old behavior.

### timed_chunk_client: two dispatch bugs fixed

Same class of bug as the streaming client had — `time_from_start` relative to the
publish instant.

1. **Speed spike at every chunk boundary.** `first_ts = floor(s_now) + 1 +
   extra_delay_steps` is 1–2 steps of travel ahead of the arm, but `_send_trajectory`
   gave the first point a flat `dt`. Simulated: 1.14× every cycle at 0.45 s inference,
   2.00× after a 2.6 s stall. Fixed by passing `lead = (first_ts - s_disp) * dt`,
   recomputed at dispatch, and anchoring on `(first_ts, t_dispatch + lead)` so the
   wall-clock model matches what was actually commanded. Now 1.00× in both cases.
   `lead` is a bounded sawtooth 1.0–1.86 dt (wraps every ~7 cycles at these numbers)
   because `extra_delay_steps=1` aims at the next-*next* integer timestep — distance
   and time scale together, so the speed is right and there is no pause.
2. **Silent rewind.** `min_dispatch_steps` caps `start_idx`, so when inference outran
   the chunk the first waypoint landed *behind* the arm. Now checked against the arm's
   actual position at dispatch (`gap_steps < 0.5` → drop the chunk and re-infer, with
   a `[stale]` message naming the flags to change).

3. **Executing timestep is now measured, not estimated.** `UR5SensorNode` subscribes
   to `/scaled_joint_trajectory_controller/controller_state` (and `/…/state`, the
   older name) and `/speed_scaling_state_broadcaster/speed_scaling`. `executing_step`
   locates the controller's *desired* position inside `self.pending` — projected onto
   the segment between two waypoints, so it has sub-step resolution — and falls back
   to extrapolation when the topic is stale, absent, or the match residual exceeds
   `--ctrl-state-max-resid`. The search is windowed (`--ctrl-state-window`) because
   joint poses repeat over a task and a global nearest-waypoint match would teleport
   the estimate whenever the arm passes near an earlier pose.

   Speed scaling was the reason the old estimate could be *wrong* rather than merely
   imprecise: with `open_loop_control: true` JTC advances its clock as
   `traj_time_ += period * scaling_factor_`, so at 50% on the pendant the arm runs
   half speed while an unscaled estimate keeps counting at full rate and drifts
   without bound. Note the broadcaster publishes a **percentage**
   (`speed_scaling_state_broadcaster.cpp:140` multiplies by 100).

   `_wait_until_trigger` now computes remaining chunk time from the measured step at
   the scaled rate. Startup prints whether controller_state arrived and what the
   scaling factor is; the per-cycle line reports `scale=` and `est_fallbacks=`.

Same caveat applies to the streaming client's absolute schedule: it assumes speed
scaling is 1.0. Not wired up there yet.

Not addressed: the trigger's latency EMA (a slow inference still drains the chunk and
the arm holds — the gap that gets worse as dt shrinks), and the action-vs-topic
question. For the latter, first check the controller log for `Aborted due to path
tolerance violation`: action goals enforce `constraints.<joint>.trajectory: 0.2` from
`ur5_real_controllers.yaml`, topic-published trajectories do not, and an abort holds
position — which looks exactly like an inter-chunk gap.

### Streaming client is production; measured-best values are now the defaults

| flag | was | now | why |
|---|---|---|---|
| `control_interface` | forward_position | `jtc_topic` | controller interpolates; no dense Python stream to keep fed |
| `aggregate_fn_name` | conservative | `ramp` | commanded rms jerk 38.9 -> 4.5 |
| `chunk_size_threshold` | 0.5 | `0.2` | overlap 17 -> 7 steps to blend |
| `lookahead` | 4 | `30` | 1.5 s of runway instead of 0.2 s |
| `jtc_end_velocity` | zero | `continue` | stops commanding a stop on every replan |

`action_horizon` stays 16 (GR00T's chunk size). Old behavior still reachable per flag
for A/B.

**Speed scaling is now honored** (`--use-speed-scaling`, default on). `_step_period()`
returns `dt / scaling_factor` and feeds BOTH the tick period and the published
schedule. Scaling only the schedule would keep popping the queue at full rate and
outrun the arm; scaling only the period would hand the controller a schedule it cannot
meet. The anchor is rebuilt whenever the factor moves, since the anchor encodes a rate.
Verified: factor 1.0 reproduces the old times exactly, 0.5 stretches everything 2x
(so still 1.00x commanded speed), a 0.0 reading is clamped rather than dividing by
zero, and the topic never arriving leaves the factor at 1.0. Startup warns if it is
not 1.0.

Measured state at handover (dt 0.05, both clients):

| | streaming | timed_chunk |
|---|---|---|
| commanded rms jerk | **4.5** | 30.0 |
| commanded max jerk | **64** | 191 |
| tracking rms residual | 9.57 mrad | **1.67** |
| tracking lag | **0 ms** | +53 ms |
| seam artifact | none measurable | stall-then-lurch, wp0 travel 0.0026 vs 0.0097 rad |

timed_chunk's seam is a *velocity* notch, not a position step — `boundary_blend` does
not touch it (tested: `--no-boundary-blend` moved p90 seam jerk 178.7 -> 165.6 only),
and `ramp` cannot fix it with only 3 overlapping timesteps to work with. Untested
remedy: `--trigger-margin-steps 8` to grow the overlap.

Streaming's remaining weakness is tracking residual, and it is not the trajectory —
its command is 6x smoother than timed_chunk's yet tracked 4x worse. That points at the
20 Hz republish and the `tick_err` p95 of 22 ms from whole-process stalls.

### Next

- [ ] Re-run with the absolute schedule; check the drop count and whether the
      commanded-vs-measured wobble at 14 Hz goes away.
- [ ] `--aggregate_fn_name ramp` for the merge-aligned jerk spikes: with
      `chunk_size_threshold 0.4` they got *worse*, p90 jerk 51.0 within 250 ms of a
      merge vs 12.7 elsewhere (fewer seams, but each blends 18 steps of a staler
      prediction). Between-seam roughness did improve, 15.0 -> 12.7.
- [ ] `gc.disable()` experiment for the 25 ms whole-process stalls.
- [ ] `js` at 35 Hz makes every measured-side number unusable; if it stays there after
      the stall fix, record `/joint_states` out of process (`ros2 bag`).
- [ ] Read the publish-interval stats from the run just done.
- [ ] **`--stream-hz 250`** — one flag, no rebuild, directly tests mechanism 1,
      and is what teleop does.
- [ ] Confirm the driver was launched with `ur_patched.urdf.xacro`; otherwise
      `servoj_lookahead_time` is the stock 0.03, not the patched 0.1.
- [ ] For *slowness* specifically: `--dt 0.07` **is** the playback speed (14 Hz
      waypoints). Try `--dt 0.05` and see whether it tracks directly, before
      blaming filter lag (28 ms at `fc=8`, roughly half a `dt`). Also still
      untested: `--aggregate-fn-name latest_only`.

- [ ] Fill the table above (3 runs each, same task / `--chunk-filter rts` /
      `--aggregate-fn-name` / `--dt`).
- [ ] Check the `*_stream.png` velocity panel: green lines are control ticks;
      with `linear` the velocity steps at every one, with `hermite` it should
      pass through continuously.
- [ ] Confirm the added lag is acceptable — Butterworth group delay is
      ≈ `√2/(2π·fc)` = 28 ms at 8 Hz, on top of `servoj_lookahead_time`. The
      outer loop here is the VLA itself, so too much total lag makes it
      over-correct. Give lag back from `servoj_lookahead_time` before `fc`.
- [ ] Sweep `servoj_lookahead_time` 0.03 / 0.05 / 0.1 with the filter on — it is
      now partly redundant with the Butterworth. Requires editing
      `src/dualsense_teleop/urdf/ur.ros2_control.xacro:54-56` +
      `colcon build --packages-select dualsense_teleop` + driver restart, and
      only applies when launched with `description_file:=ur_patched.urdf.xacro`
      (see `launch_all.md`). Note `ur5_pose_tracking.launch.py:99` uses the stock
      `ur_description` URDF, so the teleop stack itself has never run the patched
      2000/0.1 values — don't treat them as a proven baseline.
