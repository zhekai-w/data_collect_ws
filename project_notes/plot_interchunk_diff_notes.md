# How `scripts/plot_interchunk_diff.py` Calculates Inter-Chunk Difference

## Input

Loads an `.npz` log file (produced by `ur5_gr00t_simple_client.py --log`) containing:

| Key | Shape | Description |
|-----|-------|-------------|
| `t` | `(N,)` | Timestamps for each commanded waypoint |
| `cmd` | `(N, 6)` | Raw commanded arm waypoints (6 joints: pan, lift, elbow, wrist1, wrist2, wrist3) |
| `chunk` | `(N,)` | Chunk/cycle ID for each waypoint |
| `js` | `(M, 7)` | *(optional)* High-rate `/joint_states` log: column 0 = time, columns 1-7 = joint positions in driver order `[lift, elbow, w1, w2, w3, pan]` |

## Core Calculation

### 1. Chunk Boundary Detection

```python
boundary_idx = np.flatnonzero(np.diff(chunk) != 0) + 1
```

- `np.diff(chunk)` produces the difference between consecutive chunk IDs.
- Non-zero entries mark where one chunk ends and the next begins.
- `+1` shifts the index to point at the **first waypoint of the new chunk** (the last waypoint of the old chunk is at `boundary_idx - 1`).

### 2. Inter-Chunk Difference

```python
boundary_diff = cmd[boundary_idx] - cmd[boundary_idx - 1]  # (num_boundaries, 6)
mag = np.abs(boundary_diff)
```

- For each chunk boundary, subtracts the **last commanded waypoint of chunk N-1** from the **first commanded waypoint of chunk N**.
- This is the position discontinuity (jump) that `boundary_blend` is designed to smooth.
- `mag` is the element-wise absolute value, giving per-joint jump magnitude.

### 3. Measured Position (optional)

If `js` key exists with sufficient data:
- Reorders columns from driver order `[lift, elbow, w1, w2, w3, pan]` to display order `[pan, lift, elbow, w1, w2, w3]` via `jp[:, [5, 0, 1, 2, 3, 4]]`.
- Interpolates measured positions onto the commanded timestamps using `np.interp`.

## Output

### Plot (`*_interchunk.png`) — 7 panels:

| Panel | Content |
|-------|---------|
| 1–6 | Per-joint commanded (blue `-o`) and measured (orange line) position vs. time, with gray dashed vertical lines at chunk boundaries |
| 7 | Per-joint `|Δ|` (inter-chunk jump magnitude) at each boundary vs. time |

### Console summary:

- Number of chunk boundaries found
- Per-boundary: timestamp, chunk IDs (N-1 → N), max `|Δ|` joint and value
- Per-joint mean `|Δ|`
- Overall mean of max-per-boundary `|Δ|`, median, and global max with joint name
