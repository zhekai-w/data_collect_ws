# How Joint Jerk Is Calculated

Reference implementation: `pkgs/Isaac-GR00T/scripts/eval_policy.py`
(reused in `pkgs/Isaac-GR00T/scripts/plot_jerk_filter.py`).

## What jerk is

Jerk is the **third time-derivative of position** — the rate of change of
acceleration. For a joint trajectory `q(t)`:

```
jerk(t) = d³q/dt³
```

High jerk means the motion is snapping between accelerations (jittery,
mechanically stressful). Filtering a trajectory should reduce jerk.

## Discrete computation

Given a trajectory array `traj` of shape `(T, D)` — `T` timesteps, `D` joints —
and a fixed timestep `dt` (seconds), jerk is approximated with a 3rd-order
finite difference along the time axis:

```python
jerk = np.diff(traj, n=3, axis=0) / (dt ** 3)   # shape (T-3, D)
```

- `np.diff(..., n=3, axis=0)` applies the first difference three times along
  time. Expanded, each element is:

  ```
  Δ³q[t] = q[t+3] - 3·q[t+2] + 3·q[t+1] - q[t]
  ```

- Dividing by `dt**3` converts the unitless difference into physical units
  (rad/s³ for revolute joints).
- The result has `T-3` rows: three samples are consumed by the triple
  difference, so jerk is only defined from the 4th sample onward. When plotting
  against time, align it with `t[3:]`.

**Assumption:** `dt` is constant. If timestamps are irregular, this uniform-grid
formula is only approximate. In `plot_jerk_filter.py`, `dt` is taken as the
**median** of `np.diff(t)` from the log.

## Aggregate metric: RMS jerk

To collapse the whole `(T-3, D)` jerk field into a single figure of merit:

```python
def compute_rms_jerk(traj, dt):
    jerk = np.diff(traj, n=3, axis=0) / (dt ** 3)
    return float(np.sqrt(np.mean(jerk ** 2)))
```

`np.mean(jerk**2)` has **no `axis`**, so it averages over *every* element —
all timesteps **and** all joints. This is the root-mean-square over the entire
arm:

```
RMS jerk = sqrt( (1 / ((T-3)·D)) · Σ_t Σ_j  jerk[t,j]² )
```

Notes:
- It is **RMS**, not a plain mean: values are squared before averaging, so large
  jerks dominate.
- Because all joints are pooled equally, the high-amplitude joints (e.g. lift,
  elbow) dominate the number; low-amplitude wrist joints barely move it.

### Per-joint variant

To get one RMS value **per joint** instead of a single arm-wide number, keep the
time axis only:

```python
rms_per_joint = np.sqrt(np.mean(jerk ** 2, axis=0))   # shape (D,)
```

This shows which joints a filter helps most.

## Per-timestep magnitude (for plotting)

Two ways to visualize jerk over time:

- **Cross-joint norm** (single curve, as in `eval_policy.py`):

  ```python
  jerk = np.diff(traj, n=3, axis=0) / (dt ** 3)
  j_mag = np.linalg.norm(jerk, axis=1)   # shape (T-3,)
  ```

- **Per-joint** (one curve per joint, as in `plot_jerk_filter.py`):

  ```python
  j_per_joint = np.diff(traj, n=3, axis=0) / (dt ** 3)   # shape (T-3, D)
  # plot np.abs(j_per_joint[:, j]) for each joint j
  ```

Both are typically plotted on a **log y-axis** because jerk spans several orders
of magnitude.

## Before vs. after filtering

To quantify how much a smoother (One Euro / Savitzky-Golay / RTS) reduces jerk:

```python
rms_raw  = compute_rms_jerk(raw_traj,  dt)
rms_filt = compute_rms_jerk(filt_traj, dt)
reduction = 100 * (rms_raw - rms_filt) / rms_raw   # percent
```

Example (One Euro filter, from `plot_jerk_filter.py`):

| log     | RMS jerk raw | filtered | reduction |
|---------|-------------:|---------:|----------:|
| gr00t   |        59.81 |    12.28 |     79.5% |
| pi0     |         7.52 |     1.59 |     78.8% |
| smolvla |        28.75 |     5.90 |     79.5% |
