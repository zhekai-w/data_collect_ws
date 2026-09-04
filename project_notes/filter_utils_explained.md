# `scripts/filter_utils.py` — Filter/Smoother Reference

Covers the two trajectory-smoothing methods in `filter_utils.py`: Savitzky-Golay and RTS (Rauch-Tung-Striebel) Kalman smoother. (OneEuroFilter and `blend_chunk_boundary` are separate utilities, not detailed here.)

## Savitzky-Golay filter (`savgol_chunk`)

Fits a low-order polynomial to a sliding window of points via least squares, and replaces the center point with the polynomial's value there. Smooths noise while preserving local curvature better than a moving average.

**Math:**

For a window of `window_length` points centered at `t=0`, fit:

```
p(t) = a0 + a1*t + a2*t^2 + a3*t^3   (polyorder = 3)
```

by minimizing:

```
sum_i ( y_i - p(t_i) )^2
```

over the window. The smoothed output at the center is `p(0)`. In practice this reduces to a fixed convolution kernel (closed-form Gram/Savitzky-Golay coefficients), applied per-joint along the time axis.

**Parameters:**

| Param | Default | Meaning |
|---|---|---|
| `window_length` | 7 | Points per fit window. Larger → smoother but more lag/distortion at sharp turns. Must be odd and > polyorder. |
| `polyorder` | 3 | Degree of fitted polynomial. Higher → tracks curvature more closely, less smoothing, more risk of fitting noise. |
| `axis` | 0 | Applied along time axis of `(H, N)` array — H timesteps, N joints. |

## RTS smoother (`rts_smoother_chunk`)

A two-pass Kalman smoother: forward Kalman filter pass, then backward RTS pass. Because it uses the *entire* trajectory (past and future) at every point, it is optimal (minimum variance) given the noise model — unlike a causal Kalman filter, which only uses past data.

**Model:** constant-velocity per joint. State `x_t = [position, velocity]^T`.

**Forward pass** (standard Kalman filter), for `t = 1..H`:

```
Predict:  x̂_t^- = F x̂_{t-1}         P_t^- = F P_{t-1} F^T + Q
Update:   S_t = H P_t^- H^T + R      K_t = P_t^- H^T S_t^-1
          x̂_t = x̂_t^- + K_t (z_t - H x̂_t^-)
          P_t = (I - K_t H) P_t^-
```

**Backward RTS pass**, for `t = H-1 .. 1`:

```
P_pred = F P_t F^T + Q
G_t    = P_t F^T P_pred^-1
x̂_t^s  = x̂_t + G_t (x̂_{t+1}^s - F x̂_t)
P_t^s  = P_t + G_t (P_{t+1}^s - P_pred) G_t^T
```

**Matrices used in code:**

- `F = [[1, dt], [0, 1]]` — state transition (constant-velocity model).
- `H = [1, 0]` — observation matrix; only position is measured, not velocity.
- `Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]` — process noise covariance (discretized white-noise-acceleration model).
- `R = [[r]]` — scalar measurement noise covariance.

**Parameters:**

| Param | Default | Meaning |
|---|---|---|
| `dt` | 0.15 | Timestep (seconds) between waypoints. Must match the actual chunk sample rate, or the velocity state is wrong. |
| `q` | 1e-3 | Process noise. Higher → trusts the constant-velocity model less, follows raw data more closely (less smoothing, more responsive). |
| `r` | 1e-4 | Measurement noise. Higher → trusts the data less, smooths more heavily (more lag). |

The ratio `q/r` controls overall smoothing strength. With `q=1e-3 > r=1e-4`, the model moderately trusts the data over the dynamics prior.

Only the position component of the smoothed state is returned per joint (`smoothed[:, j] = sm[:, 0]`).

## S-G vs RTS

- **S-G**: local polynomial fit, no dynamics model, 2 tunable params, cheap to compute.
- **RTS**: stateful physical model (position + velocity), globally optimal under stated noise assumptions, 3 tunable params (`dt`, `q`, `r`).

RTS is preferable when velocity structure matters — e.g. `blend_chunk_boundary` needs an estimated end-velocity (`prev_end_vel`), which an RTS state naturally provides; S-G does not track velocity explicitly.
