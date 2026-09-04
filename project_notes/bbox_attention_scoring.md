# Bounding-Box Scoring of Text→Image Attention

How `scripts/visualize_text_image_cross_attention.py` + `scripts/attention_bbox.py`
turn "does the model look at the object?" from an eyeball judgement into a number.

An open-vocabulary detector locates the queried object, its box is projected onto
the ViT patch grid, and the attention mass falling inside is scored as a
**concentration ratio**. Results are drawn on the heatmaps and appended to a CSV.

---

## The pipeline

### 1. Image → patch tokens

The Eagle2.5 processor tiles the frame dynamically (a 640×480 frame becomes 12
tiles) and appends a **thumbnail** — the whole image plain-resized to 224×224
(`image_processing_eagle2_5_vl_fast.py:326-329`). Each tile is 256 tokens: a
16×16 grid of 14px patches, row-major (top-left → bottom-right).

The script scores only the thumbnail tile:

```python
thumbnail_positions = image_positions[-num_tokens_per_tile:]
```

This matters for correctness. The thumbnail is a **plain resize** — no letterbox,
no pad, no crop (`pad_during_tiling: false`, `do_pad: false` in
`preprocessor_config.json`). So pixel → patch is a pure linear scale with **no
offset**:

```
col = x / W * 16      row = y / H * 16
```

Had the processor padded to square, every box would be silently skewed.

Guards in `process_image` assert `num_img_tokens % num_tokens_per_tile == 0` and
report the tile count. When an image resolves to a single tile no thumbnail is
appended, but that tile is still a plain resize of the whole frame, so the
mapping holds.

### 2. Forward pass

Eager attention with `output_attentions=True`. For every text token of the query,
its attention row is sliced down to the 256 thumbnail columns, giving
`(layers, heads, tokens, 256)`, then averaged over the selected layers and heads.

**GR00T keeps only 12 LLM layers.** The checkpoint sets `select_layer: 12` and
`eagle_backbone.py:58-59` pops the rest of Qwen3's 28. So:

- `--layers 4` (default) = layers **9–12**
- `--layer_range 5-8` = middle layers
- `--layer_range 1-12` = genuinely all of them

Head selection: `--head N` for one, `--heads 9-12` or `--heads 0,5,11` for a
subset, `--per_head` for a grid of all 16 (rows = heads, cols = tokens).

### 3. Boxes

The detector runs on the **original full-resolution frame**, before any resizing,
and returns pixel coordinates. See "Detectors" below.

### 4. Box → patch weights

`box_patch_weights` computes **fractional** overlap per cell, not binary
membership. A patch spans 40px of a 640px frame, so a small object covers ~2
patches and in/out membership would quantize the metric badly. The computation is
separable, so it is an outer product of per-axis overlaps:

```python
wx = clip(min(xs[1:], x1) - max(xs[:-1], x0), 0) / (W / 16)
wy = clip(min(ys[1:], y1) - max(ys[:-1], y0), 0) / (H / 16)
weights = outer(wy, wx)          # (16, 16), each entry in [0, 1]
```

Multiple boxes of the same object combine with elementwise **max**
(`union_patch_weights`), never sum — summing would double-count overlapping
boxes, inflating `area_frac` and deflating the score.

---

## How `conc` is calculated

```python
p         = a / a.sum()          # renormalize over the 256 patches
mass_in   = (p * w).sum()        # attention mass inside the box
area_frac = w.sum() / 256        # fraction of the image the box covers
conc      = mass_in / area_frac
```

The renormalization is essential. The raw attention row is a softmax over the
**whole sequence** — system prompt, text, and all 13 tiles — so its 256 thumbnail
entries sum to something like 0.03, not 1. Without renormalizing, "mass inside
the box" would not be a fraction of anything meaningful.

Worked example, the mango box on `fruit.png`:

```
mass_in   = 0.044333    # 4.4% of the token's image attention lands in the box
area_frac = 0.029294    # the box covers 2.9% of the image
conc      = 0.044333 / 0.029294 = 1.51x
```

### Reading the number

`conc` is **attention density relative to chance**:

| value | meaning |
|---|---|
| `1.0` | uniform — the box gets exactly the share its area entitles it to. No grounding. |
| `>1` | attention concentrates on the object |
| `1/area_frac` | the ceiling — all mass inside the box |

For a box covering 2.9% of the frame the ceiling is ~34x, so 1.51x is very weak
and 11.7x is strong.

**Why divide by area.** A plain mean-inside-the-box rewards large boxes
automatically, making a mango box and a lemon box incomparable. Area
normalization removes that — which is what makes a big-apple/small-apple
comparison valid, since a physically larger box cannot win by being larger.

---

## Detectors and the negatives mechanism

`--box_source` selects the backend: `owlv2` (fast default), `gdino`, `owlvit`,
`omdet`. For uncommon nouns `--box_source gdino --box_model
IDEA-Research/grounding-dino-base` is markedly better.

### Why `--box_negatives` exists

Detector scores are CLIP-style similarities, **not calibrated probabilities**. A
rare word like "mango" peaks around 0.15 where "apple" reaches 0.80. With a
single query, every box gets a "mango" score and the highest wins — even when it
is a lemon, because nothing competes for it:

```
--box_query mango                     (no negatives)
  lemon box  @0.146   <- top-1, WRONG
  mango box  @0.112
```

Naming the other objects makes them compete in the same forward pass:

```
--box_negatives "lemon,banana,apple,green pepper,yellow pepper"
  lemon box  -> "lemon"  0.48 beats "mango"  -> dropped
  banana box -> "banana" 0.80 beats "mango"  -> dropped
  mango box  -> "mango"  0.35 wins           -> KEPT
```

Two steps make this work:

1. **Label matching** (`_label_matches`). OWLv2 returns an integer index into the
   prompt list; Grounding DINO returns the matched token span, which for a
   multi-phrase prompt merges into strings like `"mango pepper yellow pepper"`.
   Exact string equality discards correct detections, so string labels are
   matched by **word containment**.
2. **Argmax over labels** (`_wins_its_box`). gdino and omdet score *every*
   (box, label) pair rather than one label per box, so a target-labelled
   detection can coexist with a higher-scoring detection of the same object under
   another name. A target box is dropped when a competing label outscores it on
   the same box (IoU ≥ 0.7).

Finally, the target **claims its detections first** and negatives are computed
from what remains (`exclude=target_mask` in `select`). Without this, a merged
gdino span matching both "mango" and "yellow pepper" would emit the same box
twice under both names.

The negative equal to the target is dropped automatically, so one fixed list can
be reused across every query:

```bash
NEG='apple,red mango,lemon,banana,green pepper,yellow pepper'
for q in apple "red mango" banana; do
  python scripts/visualize_text_image_cross_attention.py \
    --image_path scripts/images/fruit1.png --text_query "$q" \
    --box_source gdino --box_model IDEA-Research/grounding-dino-base \
    --box_negatives "$NEG" --box_threshold 0.15 --head 10 \
    --output_dir attention_outputs/fruit1_all
done
```

### Phrasing matters

Bare nouns can lose to a competing label by a hair. On `fruit1.png` the mango
region scored `apple @0.290` vs `mango pepper @0.278`, so the argmax gave it to
apple. `red mango`, `mango fruit`, and `red and yellow mango` all take the region;
`mango` and `ripe mango` do not.

`--box_query` (detection) is separate from `--text_query` (attention). Keep
`--box_query` fixed across a comparison so the geometry is identical and only the
text varies.

---

## Output

**Plots.** Target boxes in green (numbered to match `box_idx`), distractors in
red with their label. Caption is `max attention: X | conc Yx`:

- `max attention` — the single largest raw attention weight among the 256
  patches. Not renormalized, so it is tiny (0.001–0.07). It says how peaked the
  map is, not where. A border attention sink can inflate it.
- `conc` — the concentration ratio. This is the grounded number.

**CSV**, appended to `<output_dir>/attention_scores.csv`:

```
frame, image_path, text_query, box_query, n_boxes, box_idx, label,
x0, y0, x1, y1, det_score, layers, head, token, mass_in, area_frac, concentration
```

One row per (frame, token, head, box). `box_idx` is `union` for all target
instances together, `0,1,…` per target instance, and `neg0,…` per distractor —
`box_idx` plus `label` identify a box. Every row is the **same** metric: that
text query's attention, measured on that box. The negatives are not scored with
their own name; they are just regions.

With a single detected box the `union` row duplicates the `box_idx=0` row.
Filter with `awk -F, '$6!="union"'`.

---

## Caveats

- **Scope.** This measures the LLM's text→patch attention, downstream of SigLIP,
  the connector, and 12 Qwen layers. A null result bounds the whole stack, not
  SigLIP alone.
- **Attention sinks.** A few border patches absorb much of the mass, depressing
  every `conc` in a frame equally. Compare within a frame, not across frames.
- **Subword tokens.** `mango` tokenizes to `m` + `ango`; only the latter carries
  signal. Multi-word queries give one row per token.
- **Detector output is not ground truth.** Spot-check the drawn boxes. Where the
  detector is genuinely ambiguous — it scored the mango identically as "mango"
  and "yellow pepper" — treat that box as less trustworthy.
- **The script's view ≠ the policy's view.** Full-resolution frames give 13 tiles;
  the deployed policy sees a 0.95 center-crop at 224 (1 tile, see
  `data_config.py:172-173`, `policy.py:97`). `--policy_preproc` matches the
  policy and transforms the boxes with it.
