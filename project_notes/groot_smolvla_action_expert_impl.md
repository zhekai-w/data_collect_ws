# How the GR00T N1.5 ⟵ SmolVLA action-expert swap is actually implemented

This documents the **as-built** implementation (verified against the code and against
upstream SmolVLA), not the original plan. For the design rationale see
[`pkgs/groot_smolvla_action_expert_swap.md`](pkgs/groot_smolvla_action_expert_swap.md);
this note records what the code does and where the implementation diverged from that plan.

The swap replaces GR00T N1.5's diffusion-transformer (DiT) action head with a vendored,
self-contained port of **SmolVLA's flow-matching action expert**, re-targeted to
cross-attend GR00T's single Eagle backbone feature tensor. The Eagle 2.5 VLM backbone,
data pipeline, transforms, and training scripts are untouched.

## Files

| File | Role |
|---|---|
| `pkgs/Isaac-GR00T/gr00t/model/gr00t_n1_smolvla.py` | Standalone **copy** of `gr00t_n1.py`; adds an action-head registry. |
| `pkgs/Isaac-GR00T/gr00t/model/action_head/smolvla_action_head.py` | The vendored SmolVLA-style flow-matching head. |

`gr00t_n1.py` and the original `flow_matching_action_head.py` are **not modified**.

## Wire-in (the model wrapper)

Instead of branching inside `gr00t_n1.py`, the implementation makes a full copy,
`gr00t_n1_smolvla.py`, so upstream stays clean:

- **Registry** keyed by `action_head_cfg["type"]` (`gr00t_n1_smolvla.py:50`):
  ```python
  ACTION_HEAD_REGISTRY = {
      "flowmatching": (FlowmatchingActionHead, FlowmatchingActionHeadConfig),
      "smolvla":      (SmolVLAActionHead,      SmolVLAActionHeadConfig),
  }
  ```
- In `__init__` (`:104-110`) the `type` key is popped from `action_head_cfg` (default
  `"flowmatching"` for back-compat) to select the head class + config class.
- Classes/`model_type` are renamed to `GR00T_N1_5_SmolVLA` / `gr00t_n1_5_smolvla`
  (`:64`, `:82`) with their own `AutoConfig`/`AutoModel` registration (`:265-266`) so the
  original GR00T registration is not clobbered.
- `forward` / `get_action` / `prepare_input` / `validate_data` are **unchanged** from
  `gr00t_n1.py` — the new head satisfies the same contract, so the seam
  (`backbone → action_head`) does not move.

Because the backbone runs once and hands a single `backbone_features (B, S, 1536)` tensor
to the head, this is a **decoupled** swap — no `EagleBackbone` surgery.

## The action head (`smolvla_action_head.py`)

A self-contained ("vendored") port. It does **not** import lerobot's
`SmolVLMWithExpertModel`; the transformer block is re-implemented locally.

### Contract (matches `FlowmatchingActionHead`)
- `prepare_input(batch) -> BatchFeature` — pass-through (`:283`).
- `forward(backbone_output, action_input) -> {"loss"}` (`:344`).
- `get_action(backbone_output, action_input) -> {"action_pred": (B, H, action_dim)}` (`:372`).
- `set_trainable_parameters(tune_projector, tune_diffusion_model)` (`:253`),
  `process_backbone_output` (`:303`), `dtype`/`device` properties (`:286-292`).

### Submodules (single-robot SmolVLA flavor)
- `vlln = LayerNorm(1536)` — preprocesses backbone features into cross-attn **memory** (`:221`).
- `state_proj = Linear(max_state_dim, H)` — state enters the **expert** as a prepended
  token (the backbone already ran, so state cannot go into the VLM prefix as upstream
  SmolVLA does) (`:224`, `:335-341`).
- `action_in_proj = Linear(action_dim, H)`; `action_time_mlp_in/out` fuse action with a
  sinusoidal-τ embedding (`:225-227`, `:306-320`).
- `action_out_proj = Linear(H, action_dim)` (`:228`).
- **Expert transformer**: `num_layers` × `ExpertBlock`, each = RMSNorm + attention +
  SwiGLU MLP, residual (`:113-172`, `:231-248`). Layer `idx` does **causal RoPE
  self-attn** if `self_attn_every_n_layers > 0 and idx % self_attn_every_n_layers == 0`,
  otherwise **cross-attn** to backbone memory (per-block `k_proj/v_proj: 1536 → H`)
  (`:234-237`, `:133-135`). Single-robot: plain `nn.Linear`, no `CategorySpecificMLP`;
  `embodiment_id` is ignored.

### Token layout & forward flow
`_build_tokens` → `[state_token, action_0 … action_{H-1}]` (length `H+1`).
`_run_expert` runs all layers, `final_norm`, then returns the **last `H` tokens**
(`:322-333`) — dropping the state token — which `action_out_proj` maps to velocity.

### Flow matching (SmolVLA convention — verified identical to upstream)
- Training (`:353-367`): `x_t = t·noise + (1−t)·actions`, target `u_t = noise − actions`,
  predict `v_t`, loss `MSE(v_t, u_t)`.
- Time sampling (`:295-301`): `Beta(1.5, 1.0)`, then `t = s·0.999 + 0.001`.
- Inference (`:385-393`): start `x_t = 𝒩(0,1)`, `dt = −1/num_steps`, Euler
  `x_t += dt·v_t` for `num_steps`, `t: 1 → 0`.

These match `lerobot/.../modeling_smolvla.py` (`forward` L840-841, `sample_actions`
L891-903, `sample_time` L687-689) exactly.

## Config (`SmolVLAActionHeadConfig`, `:178-207`)

Defaults: `backbone_embedding_dim=1536`, `expert_hidden_size=720`, `num_layers=16`,
`num_heads=12`, `self_attn_every_n_layers=2`, `num_steps=10`, `min_period=4e-3`,
`max_period=4.0`, `noise_beta_alpha=1.5`, `noise_beta_beta=1.0`, `time_sampling_s=0.999`,
`use_vlln=True`, `add_state_token=True`, `model_dtype="float32"`, `tune_projector=True`,
`tune_diffusion_model=True`. `action_dim`/`action_horizon`/`max_state_dim`/`max_action_dim`
default to `None` and must be supplied at construction.

## Deviations from the plan / things to watch

1. **Wire-in is a copied model file + registry**, not an in-place edit of `gr00t_n1.py`
   as the plan's Part 2 described. Functionally equivalent and cleaner.
2. **No `max_num_embodiments`** config field (consistent with the single-robot decision —
   embodiment ignored). Added fields not in the plan: `time_sampling_s`, `use_vlln`,
   `add_state_token`, `model_dtype`.
3. **`action_horizon=50` is not a default** — the chunk-50 decision must be enforced by
   the config passed at construction (only `expert_hidden_size=720` is baked in).
4. **Loss adds `action_mask` weighting** on top of upstream's plain `MSE(u_t, v_t)`
   (`:365-367`) — sensible for padded action dims; not mentioned in the plan.
5. **Cosmetic**: `from_pretrained` still prints "Tune action head DiT"
   (`gr00t_n1_smolvla.py:237`) though the head is now the SmolVLA expert.
6. Weights are **trained from scratch** (random init on a frozen/finetuned Eagle
   backbone); SmolVLA's pretrained expert weights are not transferable.
