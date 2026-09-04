# Visualize Text-Image Cross Attention in Eagle VLM

This document explains how `scripts/visualize_text_image_cross_attention.py` works,
from the Eagle VLM architecture perspective.

---

## How Attention Scores Are Calculated

Inside each LLM decoder layer, standard scaled dot-product attention is computed:

### 1. Query, Key, Value Projections

Each token's hidden state `h` (2048-dim) is projected into three vectors:

```
Q = h @ W_Q    # (seq_len, num_heads × head_dim)
K = h @ W_K    # (seq_len, num_heads × head_dim)
V = h @ W_V    # (seq_len, num_heads × head_dim)
```

These are reshaped per head:
- Q: `(num_heads, seq_len, head_dim)` where `head_dim = 128`
- K: `(num_kv_heads, seq_len, head_dim)` where `num_kv_heads = 8` (GQA)
- V: `(num_kv_heads, seq_len, head_dim)`

With Grouped Query Attention (GQA), every 2 query heads share 1 KV head.

### 2. Attention Score Computation

For each head, the raw attention scores are:

```
scores = Q @ K^T / sqrt(head_dim)
       = (num_heads, seq_len, seq_len)
```

`sqrt(head_dim) = sqrt(128) ≈ 11.3` scaling prevents large values from saturating softmax.

### 3. Causal Mask

A lower-triangular mask is applied so token `i` can only attend to positions `0..i`:

```
scores[mask == 0] = -inf
```

### 4. Softmax Normalization

```
attn_weights = softmax(scores, dim=-1)
```

Each row now sums to 1.0 — it's a probability distribution over previous tokens.

### 5. Output

```
output = attn_weights @ V    # (num_heads, seq_len, head_dim)
```

The **attention weights** (`attn_weights`) are what this script extracts — they tell
us how much each token "attends to" (i.e., focuses on) every other token.

---

## Eagle VLM Architecture Overview

Eagle is a Vision-Language Model (VLM) built on top of a Qwen3 LLM decoder. It has
two main components:

1. **Vision Encoder (SigLIP ViT)** — processes image tiles into patch-level embeddings
2. **LLM Decoder (Qwen3-1.7B)** — processes interleaved text + image token sequences
   through causal self-attention

### Key Insight: No Separate Cross-Attention

Eagle does **not** have a dedicated cross-attention module. Instead, image and text
tokens are **interleaved in the same sequence** and processed through standard causal
self-attention. Cross-modal understanding happens naturally: when a text token computes
its attention over positions `0..i`, it attends to whatever image tokens appear before
it in the sequence.

This script extracts those attention weights to visualize which image patches each
text token attends to.

---

## Pipeline: From Raw Image + Text to Attention Maps

### Step 1: Image Tiling & Tokenization

The image processor performs **dynamic tiling**:

- The image is resized and split into `N` square tiles (each 224×224), where `N`
  depends on the aspect ratio (1–12 tiles allowed).
- When multiple tiles are used, a **thumbnail** (entire image resized to 224×224) is
  appended for global context.
- Each tile produces `tokens_per_tile = 256` image tokens (16×16 patch grid from
  patch_size=14).

The processor generates placeholder text:
```
<img><IMG_CONTEXT> × 256 × num_tiles</img>
```

After tokenization, each `<IMG_CONTEXT>` becomes token ID **151669** (`image_token_index`).

### Step 2: Vision Encoder (SigLIP ViT)

Each 224×224 tile is processed by the ViT:

```
Input:   (B_tiles, 3, 224, 224)
    ↓ Patch Embedding (14×14 patches, linear projection to 1152-dim)
    ↓ 27 layers of bi-directional self-attention (16 heads)
Output:  (B_tiles, 256, 1152)   — 256 patch tokens per tile
```

### Step 3: MLP Connector

A linear projection maps ViT hidden size to LLM hidden size:

```
Linear(1152 → 2048)
```

Output: `(B_tiles, 256, 2048)` per tile.

### Step 4: Token Interleaving

The LLM's embedding layer produces text embeddings of shape `(B, seq_len, 2048)`.
Every position where `input_ids == 151669` (the `<IMG_CONTEXT>` token) is replaced
with the corresponding ViT image embedding:

```python
selected = input_ids == 151669
input_embeds[selected] = vit_embeds.reshape(-1, 2048)
```

The resulting sequence contains both text and image embeddings interleaved:
```
[SYS_TOKENS...] [IMG×256] [TEXT_TOKENS...]
```

### Step 5: LLM Decoder Forward Pass

The full sequence passes through 28 layers of causal self-attention:

```
Layer attention shape: (B, 16_heads, seq_len, seq_len)
```

The full Qwen3-1.7B has 28 layers, but the `EagleBackbone` truncates this to **12
layers** by default (`select_layer = -1` in the config removes all but the last 12
layers). So when the script uses `--layers 4`, it averages layers **9–12** (1-indexed).

Each layer's attention matrix tells us how much each token (row) attends to every
other token (column). Since attention is **causal**, token `i` only attends to
positions `0..i`.

---

## How the Script Extracts Attention

### 1. Load Model with Eager Attention

The script patches the model config to use `"eager"` attention implementation instead
of optimized kernels (FlashAttention, etc.). This forces PyTorch to return the full
attention weight matrices:

```python
config._attn_implementation = "eager"
outputs = model(**inputs, output_attentions=True, return_dict=True)
```

`outputs.attentions` is a tuple of 28 tensors, each of shape `(1, 16, seq_len, seq_len)`.

### 2. Identify Token Positions

The script locates:
- **Image token positions**: where `input_ids == 151669`
- **Text token positions**: all non-image tokens
- **Query token positions**: specifically the tokens matching `--text_query`

For the image, only the **thumbnail tile** (last 256 image tokens) is used, since it
represents the full image at a global scale.

### 3. Extract Cross-Attention Scores

For each selected layer, the script extracts attention from each text token to the 256
thumbnail image patches:

```python
# layer_attn shape: (16_heads, seq_len, seq_len)
head_scores = layer_attn[:, text_token_idx, thumbnail_positions]  # (16, 256)
```

This gives a `(layers, heads, text_tokens, 256)` tensor — the raw cross-attention
scores from each text token to each image patch, per head, per layer.

### 4. Averaging Modes

| Flag | Behavior | Output Shape |
|------|----------|-------------|
| *(default)* | Average across layers **and** heads | `(text_tokens, 256)` |
| `--per_head` | Average across layers only | `(heads, text_tokens, 256)` |
| `--head N` | Select single head, average layers | `(text_tokens, 256)` |
| `--heads "9-12"` | Select specific heads, average layers | `(text_tokens, 256)` or per-head |

### 5. Visualization

The 256 attention scores are reshaped to a `16×16` spatial grid, normalized, and
rendered as a color heatmap overlaid on the original image using the `cool` colormap.

Each heatmap shows which image patches the corresponding text token "looks at."

**Annotations:**
- **Black rectangle outline**: marks the patch with the highest attention score
- **Score text below**: displays the max attention value (0.000–1.000)
- **Column titles**: the decoded text token (e.g., "banana", "red", "cup")
- **Row labels** (per-head mode): head index (H0, H1, ...)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                    │
│  Image (H×W×3)  +  Text prompt ("pick up the red cup")         │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
       ┌───────▼────────┐            ┌────────▼────────┐
       │  Image Tiler   │            │   Tokenizer     │
       │  (224×224 tiles│            │                 │
       │   + thumbnail) │            │  <IMG_CONTEXT>  │
       └───────┬────────┘            │  × 256 per tile │
               │                     └────────┬────────┘
       ┌───────▼────────┐                     │
       │  SigLIP ViT    │                     │
       │  27 layers     │                     │
       │  16 heads      │                     │
       │  256 patches   │                     │
       └───────┬────────┘                     │
               │                              │
       ┌───────▼────────┐                     │
       │  MLP Connector │                     │
       │  1152 → 2048   │                     │
       └───────┬────────┘                     │
               │                              │
               ▼                              ▼
       ┌──────────────────────────────────────────────┐
       │         TOKEN INTERLEAVING                    │
       │  Replace <IMG_CONTEXT> positions with         │
       │  ViT embeddings in the text embedding        │
       │                                               │
       │  [SYS][IMG×256][TEXT_TOKENS]                  │
       └──────────────────┬───────────────────────────┘
                          │
               ┌──────────▼──────────┐
               │   Qwen3 LLM (28L)  │
               │   16 query heads    │
               │   8 KV heads (GQA)  │
               │   Causal attention  │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  output_attentions  │
               │  28 × (16, N, N)    │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  THIS SCRIPT        │
               │  Extract text→image │
               │  attention from     │
               │  last N layers      │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  16×16 heatmap      │
               │  overlaid on image  │
               └─────────────────────┘
```

---

## CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `nvidia/GR00T-N1.5-3B` | Path to Eagle model checkpoint |
| `--image_path` | `scripts/frame_000000.png` | Image, directory, or video file |
| `--text_query` | *(required)* | Text prompt to visualize |
| `--layers` | `4` | Number of final LLM layers to average over (default: last 4 of Eagle's 12 layers, i.e. layers 9–12) |
| `--per_head` | `false` | Show per-head grid instead of head-averaged |
| `--head` | `None` | Visualize a single specific head index |
| `--heads` | `None` | Average specific heads, e.g. `"9-12"` or `"0,1,2,3"` |
| `--frame_stride` | `30` | For video: sample every Nth frame |
| `--output_dir` | `attention_outputs/position_info/cube/` | Output directory for saved PNGs |

---

## Example Commands

```bash
# Default: average all heads, last 4 layers
python scripts/visualize_text_image_cross_attention.py \
    --image_path scripts/frame_000000.png \
    --text_query "pick up the red cup"

# Per-head visualization
python scripts/visualize_text_image_cross_attention.py \
    --image_path scripts/frame_000000.png \
    --text_query "banana" --per_head

# Only heads 9-12, averaged
python scripts/visualize_text_image_cross_attention.py \
    --image_path scripts/frame_000000.png \
    --text_query "red cup" --heads 9-12

# Video input
python scripts/visualize_text_image_cross_attention.py \
    --image_path ~/work/videos/demo.mp4 \
    --text_query "banana" --frame_stride 100 --per_head
```
