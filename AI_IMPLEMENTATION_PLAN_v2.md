# 🚀 Shoe VTO — AI Perception Implementation Plan
### Responding to: `SHOES_VTO_AI_REQUIREMENTS_v1.md`
**Version:** 1.0 · **Date:** 2026-09-13 · **Author:** Mostafa (AI / Perception) · **Status:** Draft for M1 Review

---

> [!IMPORTANT]
> This plan responds **section-by-section** to the AR SDK requirements. Every **FROZEN** contract item is acknowledged and locked. Open questions (§13) are answered at the end.

---

## 0. Key Changes From My Previous Plan

| Old Plan | New Requirement | Impact |
|---|---|---|
| 18 keypoints (my schema) | **16 keypoints** (AR schema, §6.1) | Full schema redesign |
| ONNX opset 17 | **ONNX opset 12** | Must verify all ops |
| Single model output | **Two-stage pipeline** (Det + Landmark) | Architecture change |
| 256×192 input (Stage B) | **224×224** square crop, no letterbox | Input size change |
| 320 for detector | **320×320** Stage A, **256×256** WASM fallback | Confirmed |
| Simple bbox output | bbox + 4 coarse KPs + class (left/right) | Detector richer |
| No mask | **2-channel segmentation mask** 160×160 (M4) | New head required |
| No rotation head | **6-D continuous rotation** (M3) | New head required |
| No presence/next_roi | **presence + next_roi** output (Stage B) | Self-sustaining tracking |
| No sigma/confidence | **Per-KP sigma + confidence** (M5) | Uncertainty head |
| No scale head | **Relative scale** in V1, **metric mm** in V1.1 | Scale head |
| Runtime: mobile ONNX | **onnxruntime-web, opset 12, WebGPU + WASM** | Browser target |
| 3,000 annotated frames | **15,000 annotated frames, 400 subjects** | 5× more data |

---

## 1. Runtime — Confirmed

- ✅ **ONNX opset 12** — all models will be validated on both WebGPU EP and WASM EP
- ✅ **No dynamic spatial dims** — all shapes fully static except batch
- ✅ **No in-graph NMS** — raw anchor output only
- ✅ **Sigmoid/softmax baked in-graph** on: visibility, presence, confidence, mask
- ✅ **No CPU fallback ops** — will ship op list per model (§10.8)
- ✅ **Safari = WASM only** → int8-dynamic models are mandatory deliverables

**Ops to explicitly avoid:** `NonMaxSuppression`, `RoiAlign`, `GridSample`, `ScatterND`

---

## 2. Two-Stage Pipeline Architecture

```
FULL FRAME (1920×1080)
       │
       ▼  [Stage A — runs on acquisition / loss only]
┌─────────────────────────────────────────────────────┐
│  foot_det_*.onnx                                    │
│  Input:  [1, 3, 320, 320]  RGB fp32 [0,1]           │
│  Output: [1, 18, 2100]  raw anchors (no NMS)        │
│          rows 0–3: cx, cy, w, h (letterbox pixels)  │
│          rows 4–5: class scores (left_foot, right_foot) │
│          rows 6–17: 4 coarse KPs × 3 (x, y, v)     │
│          toe_tip, heel_back, ball_medial, ball_lateral │
└──────────────────────┬──────────────────────────────┘
                       │
                       │  AR SDK: decode, pick top-2, build
                       │  rotated crop using 4 coarse KPs
                       │
                       ▼  [Stage B — runs EVERY FRAME per foot]
┌─────────────────────────────────────────────────────┐
│  foot_lmk_*.onnx                                    │
│  Input:  [1, 3, 224, 224]  square crop, no padding  │
│          (already rotation-normalized by AR side)   │
│  Outputs:                                           │
│    keypoints   [1, 16, 3]  (x, y, v) crop-norm      │
│    rotation    [1, 6]      6-D continuous rotation  │
│    scale       [1, 2]      foot_length, foot_width  │
│    scale_sigma [1, 2]      uncertainty in mm        │
│    sigma       [1, 16]     per-KP positional σ      │
│    confidence  [1, 16]     baked smoothstep [0,1]   │
│    presence    [1, 1]      "foot still in crop"     │
│    next_roi    [1, 5]      cx,cy,w,h,θ for next crop│
│    mask        [1, 2, 160, 160]  sigmoid in-graph   │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
           HANDOFF TO AR SDK
   { keypoints, rotation, mask, scale, presence, next_roi }
```

**Stage B is self-sustaining**: `next_roi` feeds directly into the next frame's crop.
Stage A re-runs **only when `presence < threshold`** (tracking lost).

---

## 3. Keypoint Schema — **FROZEN after M1**

**16 keypoints per foot** (replacing my old 18-KP schema):

| # | Name | Zone | Tier | Anatomical Definition |
|---|---|---|---|---|
| 0 | `toe_tip` | sole | 1 | Most anterior point of foot/shoe outline |
| 1 | `toe_ground` | sole | 1 | Front of sole where it meets the floor (long axis) |
| 2 | `heel_back` | sole | 1 | Most posterior point of heel outline |
| 3 | `heel_ground` | sole | 1 | Back of sole meets floor — **canonical frame origin** |
| 4 | `ball_medial` | ball | 1 | 1st metatarsal head — big-toe side widest point |
| 5 | `ball_lateral` | ball | 1 | 5th metatarsal head — little-toe side widest point |
| 6 | `ball_top` | ball | 1 | Top of forefoot, above midpoint of 4–5 |
| 7 | `instep_top` | mid | 1 | Apex of instep (highest point of foot top) |
| 8 | `arch_medial` | mid | **2** | Sole at deepest point of medial arch |
| 9 | `midfoot_lateral` | mid | **2** | Sole at lateral midfoot, opposite 8 |
| 10 | `malleolus_medial` | ankle | 1 | Centre of inner ankle bone |
| 11 | `malleolus_lateral` | ankle | 1 | Centre of outer ankle bone |
| 12 | `ankle_center` | ankle | 1 | Midpoint of ankle joint (between 10, 11) |
| 13 | `throat` | ankle | 1 | Front ankle crease / shoe tongue top |
| 14 | `achilles` | ankle | 1 | Back of ankle at Achilles, level with 12 |
| 15 | `shin_mid` | leg | 1 | Leg centreline, 120 mm proximal to 12 |

**Tier 2** = indices **8 and 9** only. All others are Tier 1.
Tier 2 may be labeled `v=0` in the first dataset pass — **tensor layout never changes**.

**Zones:**
- `sole_line` → [0, 1, 2, 3, 8, 9]
- `ball_ring` → [4, 5, 6]
- `ankle_ring` → [10, 11, 12, 13, 14]
- `leg_axis` → [12, 15]

**Visibility convention:**
- `v=0` → not labeled / outside frame → excluded from ALL loss
- `v=1` → covered (sock, trouser, shoe) → position supervised, visibility = **negative**
- `v=2` → visible and unoccluded → position supervised, visibility = **positive**

**⚠️ Critical:** Train visibility head on `v == 2` ONLY (§6.2 fix from wrist v17→v18).

---

## 4. Canonical Foot Frame — **FROZEN after M1**

```
Origin (O)  = keypoint 3 (heel_ground)
+Z (fwd)    = unit(toe_tip − heel_back), projected onto floor plane
+Y (up)     = floor normal, pointing up
+X (right)  = +Y × +Z  (right-handed)
```

- Right foot: `+X` → medial (big-toe) side
- Left foot: `+X` → lateral (little-toe) side
- **DO NOT negate any axis** to fix a mirrored foot

**Rotation output:** 6-D continuous rotation (Zhou et al.) — first two columns of R, recovered by Gram–Schmidt. No quaternions. No Euler angles.

---

## 5. Model Architecture Plan

### Stage A — Foot Detector

| Property | Value |
|---|---|
| Base model | YOLOv8n-pose (fine-tuned) |
| Input | `[1, 3, 320, 320]` RGB fp32 [0,1] NCHW |
| Output | `[1, 18, 2100]` raw anchors, **no NMS** |
| Classes | 2 — `left_foot=0`, `right_foot=1` |
| Coarse KPs | 4 × (x, y, v): `toe_tip, heel_back, ball_medial, ball_lateral` |
| WASM fallback | 256×256 variant |
| Size budget | ≤ 5 MB fp32, ≤ 2.5 MB fp16, ≤ 1.5 MB int8 |
| Time budget | ≤ 12 ms WASM |

### Stage B — Landmark + Pose + Mask + Scale

| Property | Value |
|---|---|
| Base model | MobileNetV3-Small + custom multi-head decoder |
| Input | `[1, 3, 224, 224]` square crop, RGB fp32 [0,1] NCHW, **no padding** |
| Size budget | ≤ 14 MB fp32, ≤ 7 MB fp16, ≤ 4 MB int8 |
| Time budget | ≤ 7 ms WASM per foot (≤ 16 ms for 2 feet) |

**Stage B output heads (all frozen names):**

| Tensor | Shape | When |
|---|---|---|
| `keypoints` | `[1, 16, 3]` | M2 |
| `presence` | `[1, 1]` | M2 |
| `next_roi` | `[1, 5]` | M2 |
| `rotation` | `[1, 6]` | M3 |
| `mask` | `[1, 2, 160, 160]` | M4 |
| `confidence` | `[1, 16]` | M5 |
| `sigma` | `[1, 16]` | M5 |
| `scale` | `[1, 2]` | M5 (relative), V1.1 (metric mm) |
| `scale_sigma` | `[1, 2]` | M5 |

**All heads share one backbone.** The mask decoder adds ~1.5 ms. If it exceeds budget, it will become a separate tiny model.

---

## 6. Dataset Plan

### Volume Targets

| Item | Minimum | Goal |
|---|---|---|
| Annotated frames | **15,000** | 20,000 |
| Distinct subjects | **400** | 500 |
| Distinct video clips | **800** | 1,000 |
| Still photos | ≥ 3,000 | to avoid video-only bias (§4.1) |

### Coverage Matrix (Required by §9.2)

| Axis | Buckets | My Target |
|---|---|---|
| **Distance** | 0.3–0.6 m | 25% |
| | 0.6–1.2 m | 35% |
| | 1.2–2.0 m | 25% |
| | 2.0 m+ | 15% |
| **Camera pitch** | 0–30° (steep top-down) | 30% |
| | 30–60° | 40% |
| | 60–90° (near-horizontal) | 30% |
| **Foot yaw** | 8 buckets of 45° | ≥ 8% each |
| **Footwear** | barefoot | 20% |
| | socks | 15% |
| | sneakers | 25% |
| | boots | 15% |
| | sandals | 10% |
| | heels/formal | 10% |
| | other | 5% |
| **Skin tone** | Fitzpatrick I–VI | ≥ 10% each |
| **Motion** | walking / lifting / rotating | ≥ 30% |
| **Motion blur** | visible blur | ≥ 15% |
| **Occlusion** | partial (hem, other foot, object, edge) | ≥ 20% |
| **Frame edge** | foot clipped by boundary | ≥ 10% |
| **Two feet in frame** | both feet visible | ≥ 35% |
| **Feet crossing** | overlapping feet | ≥ 8% |
| **Floor surface** | carpet, tile, wood, concrete, grass, sand, patterned, reflective | ≥ 8 types |
| **Lighting** | indoor / outdoor / **low-light** | low-light ≥ 15% |

### Data Sourcing Strategy (3 Stages)

| Stage | Source | Frames | Manual Annotation | KPs Available |
|---|---|---|---|---|
| 1 — Public pretraining | CMU + H3WB + MOOF | ~114K | ❌ None | 3 of 16 |
| 2 — Synthetic | Foot3D + Blender renderer | ~20K | ❌ Auto-labeled | 16/16 |
| 3 — Real fine-tuning | My team collection + Roboflow | **15,000** | ✅ Required | 16/16 |

### Dataset Hygiene (§9.3)
- ✅ Split by **subject** AND **clip** — never random frame split
- ✅ Perceptual-hash dedup at **phash ≤ 6** (same as wrist v20)
- ✅ Sample video clips — no consecutive frames
- ✅ Publish realized coverage matrix (not the plan — actual counts)

### Held-Out Test Set (§9.4)
- **~500 frames from AR SDK** — never trained on, never tuned against
- All §8 metrics reported on this set

---

## 7. Annotation Guide (Delivered at M1)

Full written guide with:
- Reference photo per keypoint (correct placement)
- At least 1 "common mistake" photo per keypoint
- **Special emphasis on §8.5 geometry gates:**
  - `ball_medial (4)` and `ball_lateral (5)` must be annotated **across** the foot (not along it)
  - `malleolus_medial (10)` and `malleolus_lateral (11)` same rule
- 200-frame pilot batch before bulk annotation starts

**Geometry gates I will monitor continuously:**

| Gate | Bar |
|---|---|
| Angle between ball line (4→5) and sole long axis (1→3) | median in **[75°, 105°]** |
| `\|kpt4 − kpt5\| / foot_length` near-top-down frames | median ≥ **0.30** |
| `\|kpt10 − kpt11\| / foot_length` near-top-down frames | median ≥ **0.18** |
| Overall geometry-gate pass rate | ≥ **85%** |

Run on both GT labels and model predictions. If pass rate drops below 85%, fix is in annotation guide, not loss function.

---

## 8. Training Plan (Per Milestone)

### M1 — Schema + Annotation Only (No Training)
- [ ] Finalize 16-KP schema with AR team (sign-off)
- [ ] Finalize canonical frame definition (sign-off)
- [ ] Write annotation guide (reference photos per KP + mistakes)
- [ ] Deliver `foot_canonical.json` (16 KPs in mm, reference foot = 265 mm)
- [ ] Annotate 200-frame pilot batch
- [ ] Submit pilot for AR team review (3 working day SLA)

### M2 — Detector + Basic Landmarks
**Stage A training:**
- Base: YOLOv8n-pose, COCO pretrained
- Fine-tune on foot detection dataset
- Output 4 coarse KPs (for rotated crop construction)
- Left/right class (2 classes)
- Input: 320×320, also ship 256×256 WASM variant
- Export: fp32 + fp16 + int8-dynamic ONNX opset 12

**Stage B training (landmarks only):**
- Base: MobileNetV3-Small backbone
- Heads: `keypoints [1,16,3]`, `presence [1,1]`, `next_roi [1,5]`
- Loss: Smooth-L1 on visible KPs (v>0), BCE on visibility (v=2 only), BCE on presence
- Input: 224×224 square (AR side builds the crop)
- Export: fp32 + fp16 + int8-dynamic ONNX opset 12

**Deliver also:**
- `foot_inference.js` — crop transform + exact inverse
- ONNX op list confirming WebGPU + WASM compatibility

### M3 — Rotation Head
- Add 6-D continuous rotation head to Stage B backbone
- Loss: Geodesic loss on rotation
- Target: Median geodesic error ≤ 6°, P90 ≤ 15°
- Report per-axis (yaw, pitch, roll) separately

### M4 — Segmentation Mask
- Add mask decoder to Stage B backbone
- 2-channel output [1, 2, 160, 160], sigmoid baked in-graph
- Channel 0: `foot_or_shoe` (up to ankle/kpt 12)
- Channel 1: `occluder / lower leg`
- **Channel 0 MUST include already-worn shoes** — barefoot-only mask is unacceptable
- Loss: Dice + boundary loss (to prioritize edge quality over IoU)
- Target: IoU ch0 ≥ 0.90, Boundary IoU@3px ≥ 0.75, temporal jitter ≤ 1.5 px

### M5 — Uncertainty + Scale
- Add `sigma [1,16]` and `confidence [1,16]` heads
- `confidence = 1 − smoothstep(σ, σ_lo, σ_hi)` baked in-graph
- Calibration: Spearman(σ, actual_error) ≥ +0.40
- Add `scale [1,2]` head: **V1 = relative scale** (foot_length / box_diagonal)
- Add `scale_sigma [1,2]` uncertainty

### V1.1 — Metric Scale (Separately Scoped)
- `foot_length_mm` median absolute error ≤ 5 mm, P90 ≤ 10 mm
- Ground truth: physically measured on ≥ 100 subjects, EU 34–48
- Decision: pure monocular or reference object → **answer at M1** (see §13.3)

---

## 9. Acceptance Criteria I Must Hit

### §8.1 — Detection & Keypoints
| Metric | Bar | How I Measure |
|---|---|---|
| Detection rate, all distances | ≥ 0.97 | On AR held-out set (§9.4) |
| Detection rate, 2.0–2.5 m | ≥ 0.92 | Distance-bucketed eval |
| Left/right class accuracy | ≥ 0.99 | Confusion matrix |
| Pose mAP@50 | ≥ 0.80 | COCO-style OKS metric |
| Pose mAP@50-95 | ≥ 0.62 | COCO-style OKS metric |
| NME, visible, clean | ≤ 3.0% of box diagonal | Custom eval script |
| NME, blur / occlusion | ≤ 6.0% | Stratified eval |
| Occlusion AUC | ≥ 0.80 | Binary classifier AUC on visibility head |

### §8.2 — Rotation
| Metric | Bar |
|---|---|
| Geodesic rotation error, median | ≤ 6° |
| Geodesic rotation error, P90 | ≤ 15° |

### §8.3 — Segmentation
| Metric | Bar |
|---|---|
| Mask IoU, channel 0 | ≥ 0.90 |
| Boundary IoU @ 3px, channel 0 | ≥ **0.75** |
| Mask IoU, channel 1 | ≥ 0.85 |
| Boundary temporal jitter | ≤ 1.5 px mean per consecutive frame |

### §8.4 — Temporal Stability
| Metric | Bar |
|---|---|
| KP jitter, static foot (per-KP std) | ≤ 1.0% of box diagonal |
| Spike rate (jump > 8% of box) | ≤ 2 per 100 frames |
| **Left/right class-flip rate** | ≤ **1 per 500 frames** (hard limit) |
| Re-acquisition after full occlusion | ≤ 3 frames |
| ID stability through foot-crossing | ≥ 0.98 |

> [!CAUTION]
> A class flip swaps both shoes on screen — the most damaging single failure. Treated as a **hard gate**.

> [!NOTE]
> I will **NOT** build heavy temporal smoothing into the graph. AR side applies One-Euro smoothing — I will conflict with it if I add my own.

---

## 10. All Deliverables

| # | Artifact | Milestone |
|---|---|---|
| 10.1 | `foot_det_*.onnx` + `foot_lmk_*.onnx` (fp32 + fp16 + int8) | M2–M5 |
| 10.2 | `foot_meta.json` — full sidecar per Appendix A skeleton | M2 (partial), M5 (complete) |
| 10.3 | `foot_inference.js` — crop transform + exact inverse, plain JS | M2 |
| 10.4 | `README.md` — integration notes + changelog + metric table | Each milestone |
| 10.5 | `foot_canonical.json` — 16 KPs in mm, 265 mm reference foot + scaling rule | **M1** |
| 10.6 | Validation report — all §8 tables on §9.4 set, incl. §8.5 geometry gates | M3, M4, M5 |
| 10.7 | Golden fixture — 300-frame clip + per-frame JSON output | M2 |
| 10.8 | ONNX op list per model — confirmed on WebGPU EP + WASM EP | M2 |
| 10.9 | Realized coverage matrix (actual counts, not plan) | M3 |

**`weights_sha256` filled for every shipped file. No placeholders.**

---

## 11. My Code Structure (Updated)

```
shoes_vto/
│
├── configs/
│   ├── stage_a_det.yaml              # Stage A detector config
│   ├── stage_b_lmk.yaml              # Stage B landmark config
│   └── coverage_matrix.yaml          # Dataset coverage tracking
│
├── src/
│   ├── dataset/
│   │   ├── foot_dataset.py           # Dataset: 16 KP, v0/v1/v2 visibility
│   │   ├── augmentations.py          # Augmentation (NO left/right class flip in normal path)
│   │   ├── convert_to_schema.py      # CMU/H3WB → 16-KP COCO JSON
│   │   ├── coverage_checker.py       # NEW: verify §9.2 matrix is met
│   │   └── geometry_gate.py          # NEW: §8.5 geometry gate on GT + predictions
│   │
│   ├── models/
│   │   ├── stage_a/
│   │   │   ├── detector.py           # YOLOv8n-pose, 2-class, 4 coarse KPs
│   │   │   └── train_stage_a.py      # Training script
│   │   │
│   │   └── stage_b/
│   │       ├── backbone.py           # MobileNetV3-Small shared backbone
│   │       ├── heads/
│   │       │   ├── landmark_head.py  # keypoints [1,16,3] + presence + next_roi
│   │       │   ├── rotation_head.py  # 6-D continuous rotation [1,6]
│   │       │   ├── mask_head.py      # 2-channel mask [1,2,160,160]
│   │       │   ├── sigma_head.py     # sigma [1,16] + confidence [1,16]
│   │       │   └── scale_head.py     # scale [1,2] + scale_sigma [1,2]
│   │       ├── foot_landmark_model.py # Full Stage B model (all heads)
│   │       └── train_stage_b.py      # Multi-head training script
│   │
│   ├── losses/
│   │   ├── landmark_loss.py          # Smooth-L1 on visible KPs
│   │   ├── visibility_loss.py        # BCE on v==2 only (NOT v!=0)
│   │   ├── rotation_loss.py          # Geodesic loss
│   │   ├── mask_loss.py              # Dice + boundary loss
│   │   └── sigma_loss.py             # Calibration loss
│   │
│   ├── geometry/
│   │   ├── canonical_frame.py        # 6-D rotation, Gram-Schmidt recovery
│   │   ├── foot_canonical.py         # 16 KPs in mm, 265mm reference foot
│   │   └── crop_transform.py         # Rotated crop builder + exact inverse
│   │
│   └── export/
│       ├── export_stage_a.py         # → ONNX opset 12, fp32/fp16/int8
│       ├── export_stage_b.py         # → ONNX opset 12, all heads, all dtypes
│       ├── generate_meta_json.py     # Generate foot_meta.json sidecar
│       └── generate_canonical_json.py # Generate foot_canonical.json
│
├── tools/
│   ├── evaluation/
│   │   ├── eval_detection.py         # mAP, class accuracy, distance buckets
│   │   ├── eval_keypoints.py         # NME, OKS, mAP@50/50-95
│   │   ├── eval_rotation.py          # Geodesic error per axis
│   │   ├── eval_mask.py              # IoU, Boundary IoU, temporal jitter
│   │   ├── eval_temporal.py          # Jitter, spike rate, flip rate, re-acq
│   │   ├── eval_geometry_gate.py     # §8.5 ball/malleoli angle + dist gates
│   │   └── benchmark_wasm.py         # Latency: Stage A + Stage B budget
│   │
│   └── annotation/
│       ├── verify_coverage.py        # Check §9.2 coverage matrix
│       └── geometry_gate_check.py    # Run §8.5 gate on annotations
│
├── deliverables/
│   ├── foot_meta.json                # Sidecar (Appendix A)
│   ├── foot_canonical.json           # 16 KPs in mm
│   ├── foot_inference.js             # Reference decoder (plain JS)
│   └── annotation_guide/
│       ├── ANNOTATION_GUIDE.md
│       └── reference_photos/         # 1 correct + 1 mistake per KP
│
├── data/
│   ├── raw/                          # Raw video clips
│   ├── processed/                    # Extracted frames
│   ├── annotations/                  # COCO JSON (16 KP schema)
│   └── coverage_report.json          # Realized coverage matrix (§9.3)
│
└── requirements.txt
```

---

## 12. Milestone Execution Plan

```
M1  ── Schema + Annotation Only (NO training yet)
│       ├── Freeze 16-KP schema with AR team
│       ├── Sign off canonical frame
│       ├── Write annotation guide + reference photos
│       ├── Deliver foot_canonical.json
│       ├── Annotate 200-frame pilot
│       └── AR team reviews pilot → 3 working days
│
M2  ── Detector + Basic Landmarks
│       ├── Train Stage A (YOLOv8n-pose, 2-class, 4 coarse KPs)
│       ├── Train Stage B (backbone + landmark + presence + next_roi heads)
│       ├── Export fp32 + fp16 + int8 ONNX opset 12 (both stages)
│       ├── Deliver foot_inference.js + golden fixture (300 frames)
│       └── AR team wires pipeline → returns on-device latency
│
M3  ── Rotation + First Full Acceptance Run
│       ├── Add rotation head to Stage B
│       ├── Full §8.1 + §8.2 + §8.4 evaluation on AR held-out set
│       ├── §8.5 geometry gates on both GT and predictions
│       └── Validation report
│
M4  ── Segmentation
│       ├── Add mask head (2-ch, 160×160, sigmoid in-graph)
│       ├── Full §8.3 evaluation
│       └── Boundary IoU and temporal jitter validation
│
M5  ── Uncertainty + Scale → V1 Complete
│       ├── Add sigma + confidence heads
│       ├── Add relative scale head (foot_length / box_diagonal)
│       ├── Calibrate: Spearman(σ, error) ≥ 0.40
│       └── Complete foot_meta.json sidecar
│
V1.1 ── Metric Scale (separate scope)
        ├── Decision at M1: monocular vs reference object
        └── Target: ≤ 5 mm MAE on ≥ 100 subjects EU 34–48
```

---

## 13. Answers to Open Questions (§13)

**Q1 — Rotation head approach:**
I will implement the **direct 6-D head** on the shared Stage B backbone. If it misses the §8.2 bar after M3 evaluation, I will use it as a prior and inform the AR team to fall back to PnP. Decision will be confirmed with data at M3.

**Q2 — Mask placement:**
Mask decoder will be a **head on the shared Stage B backbone**. I will measure the ms contribution separately and report it in `foot_meta.json`. If it pushes Stage B over the 7 ms budget, I will split it into a separate tiny model and benchmark both options.

**Q3 — Metric scale:**
I will assess at M1. My initial judgment: **pure monocular** is feasible for relative scale (V1) but likely not for metric mm (V1.1) without a reference object. I will flag at M1 with a concrete test result on a small batch and propose the reference object UX path if needed. This does not affect M1–M4.

**Q4 — `max_det = 2`, bystander handling:**
When more than 2 feet are detected, pick by **confidence** (highest 2). This is simpler than proximity to centre and more robust when the user steps forward. I will document this in `foot_meta.json` → `thresholds.recommended_max_det`.

**Q5 — Crop resolution 224 vs 192:**
I will benchmark both at M2. If 192×192 hits all §8 bars and buys ≥ 2 ms headroom on WASM, I will report it and we can decide. If no meaningful gain, I stay at 224×224. The tensor name is FROZEN at M1 — only the value in the sidecar changes.

**Q6 — Boots / `shin_mid` (15):**
Boot-shaft frames **will be in the dataset plan** — specifically in the "boots 15%" footwear bucket of §9.2. `shin_mid` (15) is Tier 1 in the schema. I commit to labeling it in all frames where the lower leg is visible (standing boots, partially occluded frames). It will not be V2.

---

## 14. Critical Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| WASM budget miss (> 7ms Stage B) | Medium | Benchmark at M2; fallback = split mask to separate model |
| Left/right class flip (> 1/500) | Medium | 4 coarse KPs from Stage A provide strong prior; augment crossing scenes heavily |
| §8.5 geometry gate fail on annotations | High | Dedicated geometry gate check tool run on every pilot batch before bulk |
| opset 12 op compatibility issue | Low | Test export before bulk training; avoid `GridSample`, `ScatterND` |
| Coverage matrix shortfall | Medium | Coverage checker script runs continuously during collection |
| Occlusion AUC < 0.80 | Medium | Train visibility head on v==2 only — the §6.2 fix is already in my training code |

---

## 15. What AR SDK Provides (From §11)

| Item | Owner | When |
|---|---|---|
| Real capture clips (§9.2 axes coverage) | AR/SDK | M1 kickoff |
| 500-frame held-out test set (§9.4) | AR/SDK | Before M3 |
| Canonical frame diagram + GLB origin convention | AR/SDK + 3D | M1 kickoff |
| Annotation guide + 200-frame pilot review | AR/SDK | M1, within 3 working days |
| One-Euro smoothing constants | AR/SDK | After M2 |
| Integration harness + on-device latency feedback | AR/SDK | Continuous from M2 |

---

*Prepared for M1 review. All FROZEN items acknowledged. Reply per section number for any disagreement.*

