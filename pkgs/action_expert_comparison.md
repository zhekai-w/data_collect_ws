# Action Expert Comparison: SmolVLA vs GR00T N1.5

> **Note on versions.** The original comparison was written against the GR00T **N1** paper.
> The code in `Isaac-GR00T/` is **GR00T N1.5** (`gr00t/model/gr00t_n1.py: GR00T_N1_5`).
> Key consequence: the backbone is **Eagle 2.5 VL with a Qwen3 LLM** (`hidden_size=2048`,
> 28 layers, SigLIP vision), **not** SmolLM2-1.7B. Rows that are N1-specific (e.g.
> "SmolLM2-1.7B", "Layer 12 of 24") are flagged below as *(N1 paper)*.
> Verified against `lerobot/src/lerobot/policies/smolvla/` and `Isaac-GR00T/gr00t/model/`.

## Inputs

| Input | SmolVLA | GR00T N1.5 |
|---|---|---|
| Noised action chunk | ✅ linear projection (`action_in_proj`) | ✅ embodiment-specific MLP (`MultiEmbodimentActionEncoder`) |
| Diffusion timestep τ | ✅ sinusoidal embedding fused with action via MLP (`action_time_mlp`) | ✅ fused with noised actions in MLP **and** via AdaLN |
| Robot state q_t | ❌ baked into φ_t via VLM prefix | ✅ direct, embodiment-specific MLP (`state_encoder`) |
| Language | ❌ baked into φ_t | ❌ baked into φ_t |
| VLM features φ_t | ✅ as K, V in cross-attention | ✅ as K, V in cross-attention |

> **Correction:** SmolVLA *does* handle τ explicitly. In `embed_suffix`
> (`modeling_smolvla.py:786-812`) the noisy action is linearly projected, τ gets a
> sinusoidal embedding, the two are concatenated and passed through an MLP — the same
> mechanism as GR00T's `MultiEmbodimentActionEncoder` (`flow_matching_action_head.py:56-100`).

## Architecture

| Property | SmolVLA | GR00T N1.5 |
|---|---|---|
| Base architecture | Flow Matching Transformer | Diffusion Transformer (DiT) |
| Attention structure | Alternating SA / CA layers (`self_attn_every_n_layers=2`) | Alternating CA / SA layers (`interleave_self_attention`) |
| Self-attention mask | **Causal** (no future action leakage) | **Bidirectional** |
| Timestep conditioning | sinusoidal τ fused into action tokens (MLP) | Adaptive layer norm (AdaLN) + τ fused into action tokens |
| Embodiment-specific modules | ❌ single fixed linear proj | ✅ per-embodiment MLP encoder + decoder |
| State enters expert directly | ❌ (prefix only) | ✅ (token in DiT sequence) |

> **Correction:** Neither model pairs CA + SA *inside one block*. Each GR00T
> `BasicTransformerBlock` has exactly one attention (`attn1`,
> `cross_attention_dit.py:124`); with `interleave_self_attention` on, even layers are
> cross-attn and odd layers are self-attn — the **same alternating-layer pattern** as
> SmolVLA (`smolvlm_with_expert.py:425-455`). The real GR00T distinctives are **AdaLN
> conditioning** and the **bidirectional** mask, not block-level pairing.

## Dimensions & Scale

| Property | SmolVLA | GR00T N1.5 |
|---|---|---|
| Action expert params | ~100M | ~860M *(N1; N1.5 similar order)* |
| Expert hidden dim | 0.75 × 960 = **720** (`expert_width_multiplier=0.75`) | DiT `inner_dim` = heads × head_dim (head `hidden_size=1024` default) |
| φ_t hidden dim (from VLM) | **960** (SmolLM2-360M text backbone) | **2048** (Qwen3 LLM in Eagle 2.5) |
| VLM backbone | SmolVLM2-500M-Video-Instruct | Eagle 2.5 VL (Qwen3 LLM + SigLIP) *(N1 used SmolLM2-1.7B)* |
| VLM layer tapped | Layer **16 of 32** (`num_vlm_layers=16`, text model truncated) | `select_layer` in Eagle config *(N1 paper: layer 12 of 24)* |

## Action Generation

| Property | SmolVLA | GR00T N1.5 |
|---|---|---|
| Training objective | Flow matching, `Beta(1.5, 1.0)`, s=0.999 | Flow matching, `Beta(1.5, 1.0)`, s=0.999 |
| Action chunk size H | **50** (`chunk_size`) | **16** (`action_horizon`) |
| Denoising steps at inference | **10** (`num_steps`) | **4** (forward Euler) |
| Integration | Euler (`x_t += dt * v_t`) | Forward Euler (`actions += dt * pred_velocity`) |
| Inference time per chunk | not reported | 63.9ms (L40, bf16) *(N1 paper)* |
| Output projection | Single fixed linear layer (`action_out_proj`) | Embodiment-specific MLP decoder (`action_decoder`) |

> **Note:** Both use the *identical* noise schedule — `Beta(1.5, 1.0)` with s=0.999 —
> though with opposite time conventions (SmolVLA `x_t = t·noise + (1−t)·actions`;
> GR00T `(1−t)·noise + t·actions`).

## State Routing

| | SmolVLA | GR00T N1.5 |
|---|---|---|
| State path | state → linear proj → VLM **prefix** token → baked into φ_t → expert (via cross-attn KV) | state → embodiment MLP → **DiT sequence token** alongside action tokens |
| State as expert token | ❌ never its own token (lives in prefix KV) | ✅ q_t is a real token with its own query in SA |
| Ablation verdict | Prefix (VLM) >> Suffix (expert): +7 pts avg on LIBERO | N/A (single design choice) |

> **Nuance:** SmolVLA state is in the prefix KV cache, and the expert's self-attention
> layers concatenate that cached KV (`smolvlm_with_expert.py:264`), so action tokens *do*
> attend over state — state simply never carries its own query inside the expert. The
> substantive contrast (GR00T injects state as a first-class token) still holds.

## Key Differences Summary

| Dimension | SmolVLA | GR00T N1.5 |
|---|---|---|
| State handling | Absorbed upstream into VLM prefix | First-class token in the action expert |
| SA attention mask | Causal | Bidirectional |
| Timestep conditioning | τ fused into action tokens (MLP) | τ fused into action tokens (MLP) + AdaLN |
| Embodiment support | Single (fixed projections) | Multi-embodiment (category-specific MLPs) |
| VLM backbone | SmolVLM2-500M (SmolLM2-360M text) | Eagle 2.5 VL (Qwen3-2048) |
| Model size | Lightweight (~450M total) | Large (~2–3B total) |
| Data efficiency | High (23k episodes) | Moderate (requires large diverse data) |
| Target platform | Consumer GPU / CPU | L40 / H100 GPU |
