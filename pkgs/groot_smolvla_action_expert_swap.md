# Plan: Replace GR00T N1.5's action expert with SmolVLA's action expert

## Context

Goal: (a) understand the action-expert internals of **GR00T N1.5** (`Isaac-GR00T/`) and
**SmolVLA** (`lerobot/`), and (b) swap GR00T's diffusion-transformer (DiT) action head for
SmolVLA's flow-matching action expert, while keeping GR00T's Eagle 2.5 VLM backbone and
data pipeline.

The central finding that drives the whole plan: **the two experts couple to their VLM in
opposite ways.** GR00T's action head is a *decoupled* module that cross-attends to a single
final VLM feature tensor; SmolVLA's expert is *entangled* with its VLM, sharing per-layer
KV-cache inside one fused forward pass. You cannot lift SmolVLA's expert out wholesale — its
"input" is the SmolVLM's per-layer KV, not a tensor. So the realistic swap is to port
SmolVLA's expert *design* into a GR00T-compatible action head that attends to GR00T's
backbone features.

---

## Part 1 — Code guide (read these to understand both experts)

### GR00T N1.5 action head (decoupled DiT)
- `Isaac-GR00T/gr00t/model/gr00t_n1.py`
  - `GR00T_N1_5.__init__` (L60–87): builds `self.backbone = EagleBackbone(...)` and
    `self.action_head = FlowmatchingActionHead(...)`.
  - `forward` (L161–169) / `get_action` (L171–180): **the seam.** Backbone runs once →
    `backbone_outputs`; then `action_head(backbone_outputs, action_inputs)`. This is the
    one boundary you cut.
  - `validate_data` (L128–159): the interface contract — training head must return
    `{"loss"}`, inference head must return `{"action_pred"}` of shape
    `(B, action_horizon, action_dim)`.
- `Isaac-GR00T/gr00t/model/backbone/eagle_backbone.py`
  - `forward` (L115–133): returns `BatchFeature{"backbone_features": (B,S,1536),
    "backbone_attention_mask": (B,S)}`. `eagle_linear` projects Eagle's 2048 → 1536
    (L54). A single `select_layer` is tapped — **one tensor, not per-layer.**
- `Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py`
  - `FlowmatchingActionHeadConfig` (L103–159); `FlowmatchingActionHead` (L162–215).
  - Submodules: `state_encoder` / `action_decoder` = `CategorySpecificMLP`
    (per-embodiment), `action_encoder` = `MultiEmbodimentActionEncoder` (L56–101, fuses
    sinusoidal τ into action), `future_tokens`, `vlln`+`vl_self_attention` to pre-process
    VLM features, `position_embedding`.
  - `forward` (L270–347): builds `sa_embs = [state, future_tokens, action]` as **DiT
    queries**, cross-attends to `vl_embs` (K,V), predicts velocity, MSE vs
    `velocity = actions - noise`. Time convention: `noisy = (1-t)·noise + t·action`.
  - `get_action` (L349–405): Euler loop, `actions += dt·pred_velocity`.
  - `set_trainable_parameters` (L217–238), `prepare_input` (L260–261),
    `process_backbone_output` (L263–268) — all part of the contract to reproduce.
- `Isaac-GR00T/gr00t/model/action_head/cross_attention_dit.py`
  - `DiT` (L191–306): `inner_dim = heads·head_dim`; `interleave_self_attention` →
    even layers cross-attn (K,V = `vl_embs`), odd layers self-attn. `AdaLayerNorm`
    (L44–67) injects τ. Bidirectional self-attn mask.

### SmolVLA action expert (entangled flow-matching)
- `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`
  - `VLAFlowMatching.__init__` (L607–671): `state_proj` (state→**VLM** 960),
    `action_in_proj` (action→**expert** 720), `action_out_proj` (720→action),
    `action_time_mlp_in/out` (fuse action+τ).
  - `embed_prefix` (L692–784): images+language+**state** → VLM prefix tokens (960). State
    lives in the **VLM prefix**, not the expert.
  - `embed_suffix` (L786–827): `action_in_proj(noisy)` + sinusoidal τ via
    `create_sinusoidal_pos_embedding` → concat → `action_time_mlp` → (B,50,720). Causal
    mask over the 50 action tokens.
  - `forward` (L829–865, training) / `sample_actions` (L867–906, inference): flow
    matching, `Beta(1.5,1.0)`, `x_t = t·noise + (1-t)·action`, `u_t = noise - action`,
    10-step Euler with **negative** dt (t: 1→0).
- `lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py`
  - `SmolVLMWithExpertModel` (L61–133): builds `lm_expert` as a **scaled copy of the VLM
    text config** (`hidden_size *= 0.75` → 720, `num_hidden_layers = num_vlm_layers`).
  - `forward` (L274–430): runs VLM and expert **together, layer by layer**;
    `self_attn_every_n_layers=2` decides per layer whether the expert does self-attn or
    `forward_cross_attn_layer` (L340–387) — where expert queries (720) cross-attend the
    VLM's cached per-layer K/V (960, projected 960→720 by per-layer `k_proj`/`v_proj`).
  - **This is why it can't be lifted out:** the expert's memory is the VLM's per-layer KV
    produced inside this fused loop, not a standalone tensor.
- `lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py`: `chunk_size=50`,
  `num_steps=10`, `num_vlm_layers=16`, `self_attn_every_n_layers=2`,
  `expert_width_multiplier=0.75`, `max_state_dim=max_action_dim=32`.

### The incompatibility, in one line
GR00T gives the expert **one** feature tensor `(B,S,1536)` to cross-attend; SmolVLA's
expert consumes the VLM's **per-layer** KV `(B, prefix, 960)×L` inline. The swap therefore
re-targets SmolVLA's expert to cross-attend GR00T's single feature tensor.

---

## Part 2 — Recommended approach: "decoupled SmolVLA-style head"

Build a new action head that **keeps SmolVLA's expert design** (linear action proj +
sinusoidal-τ MLP fusion, Gemma-style decoder with alternating causal self-attn /
cross-attn, `action_out_proj`, 10-step Euler, `Beta(1.5,1.0)` flow matching) but
**honors GR00T's action-head interface** and cross-attends GR00T's `backbone_features`.
Backbone, data pipeline, training scripts, and the seam at `gr00t_n1.py:167` stay intact.

### New file: `Isaac-GR00T/gr00t/model/action_head/smolvla_action_head.py`
- `SmolVLAActionHeadConfig(PretrainedConfig)`: `action_dim`, `action_horizon`,
  `max_state_dim`, `max_action_dim`, `backbone_embedding_dim=1536`, `expert_hidden_size`
  (e.g. 720), `num_layers`, `num_heads`, `self_attn_every_n_layers=2`, `num_steps=10`,
  `min_period`/`max_period`, `noise_beta_alpha/beta`, `max_num_embodiments`, tune flags.
- `SmolVLAActionHead(nn.Module)` implementing the **exact GR00T contract**:
  - `prepare_input(batch) -> BatchFeature` (pass-through, like L260–261).
  - `forward(backbone_output, action_input) -> {"loss"}`.
  - `get_action(backbone_output, action_input) -> {"action_pred": (B,H,action_dim)}`.
  - `set_trainable_parameters(tune_projector, tune_diffusion_model)` + `dtype` property
    + `process_backbone_output` (reuse `vlln`/optional self-attn).
  - Submodules:
    - state encoder → `CategorySpecificMLP` (keep GR00T multi-embodiment) **or**
      `nn.Linear` (SmolVLA single-robot) — see Decision 2.
    - `action_in_proj = nn.Linear(action_dim, expert_hidden)`; ported
      `create_sinusoidal_pos_embedding` + `action_time_mlp_in/out`.
    - **expert transformer**: a self-contained port of SmolVLA's block — RMSNorm,
      causal self-attn over `[state?, action]` tokens, cross-attn to `vl_embs` (with a
      per-head `k_proj/v_proj: 1536→expert_hidden` for the memory), MLP; alternate
      self/cross by `self_attn_every_n_layers`. (Vendor it; avoid importing lerobot's
      VLM-bound `SmolVLMWithExpertModel` — see Decision 3.)
    - `action_out_proj = nn.Linear(expert_hidden, action_dim)` (or category-specific).
  - `forward`: encode noisy action+τ → expert queries; cross-attend `vl_embs`; take last
    `H` tokens → `action_out_proj` → predicted velocity; MSE vs target. **Pick one time
    convention and keep it internally consistent** (recommend SmolVLA's
    `x_t=t·noise+(1-t)·action`, `u=noise-action`, dt<0).
  - `get_action`: 10-step Euler from Gaussian noise.

### Wire-in (small, surgical)
- `gr00t/model/gr00t_n1.py` `__init__`: branch on a new
  `config.action_head_cfg["type"]` (or `action_head_type`) to build `SmolVLAActionHead`
  vs `FlowmatchingActionHead`. **Nothing else in `forward`/`get_action` changes** because
  the new head satisfies the same contract.
- Config composition (where `GR00T_N1_5_Config` / `action_head_cfg` is built — e.g.
  finetune script / checkpoint config): add the new head's config block. Keep
  `backbone_cfg.project_to_dim = 1536`.
- Data configs (`gr00t/experiment/data_config.py`) and transforms unchanged — the new
  head consumes the same `state` / `action` / `action_mask` / `embodiment_id` keys.

### Weights
Train the new head **from scratch** (random init) on a frozen/finetuned Eagle backbone.
SmolVLA's pretrained expert weights are **not** transferable (different KV source, 960-dim
SmolVLM memory, different layer count).

---

## Locked decisions

1. **Coupling strategy → (A) decoupled.** New head cross-attends GR00T's final
   `backbone_features (B,S,1536)`. No `EagleBackbone` surgery.
2. **State / embodiment → SmolVLA single-robot.** Single `nn.Linear` projections, no
   `CategorySpecificMLP`. `embodiment_id` from the batch is ignored. Because the backbone
   has already run, state cannot go into the VLM prefix (as upstream SmolVLA does); instead
   it enters the **expert** as a prepended token via `state_proj = nn.Linear(max_state_dim,
   expert_hidden)`.
3. **Expert internals → vendor a self-contained port.** No import of lerobot's
   `SmolVLMWithExpertModel`. Re-implement the block (RMSNorm + RoPE self-attn + cross-attn
   to memory + SwiGLU) inside Isaac-GR00T.
4. **Weights → train from scratch.** Random init; frozen (or finetuned) Eagle backbone.
5. **Dims → SmolVLA `chunk_size = 50`.** `action_horizon = 50`, `expert_hidden = 720`.
   Requires `GR00T_N1_5_Config.action_horizon = 50`, the head config `action_horizon = 50`,
   and the data config `action_indices = list(range(50))`.

---

## Verification

- **Unit/contract**: instantiate `SmolVLAActionHead`; feed dummy
  `backbone_output{(B,S,1536),(B,S)}` + `state/action/action_mask/embodiment_id`; assert
  `forward` returns scalar `loss` and `get_action` returns `(B, action_horizon,
  action_dim)` — i.e. passes `GR00T_N1_5.validate_data`.
- **Overfit**: train on a handful of episodes; confirm loss → ~0 and predicted actions
  track targets.
- **End-to-end**: run `scripts/gr00t_finetune.py --data_config so100` for a few hundred
  steps with the new head; then `Gr00tPolicy.get_action` on a sample observation and check
  action shapes/ranges. A/B against the original DiT head on identical data.
