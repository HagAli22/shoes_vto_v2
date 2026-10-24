# 👟 ARShoe M1v2 — 13-Keypoint Real-Time Foot Pose Estimation

ARShoe M1v2 is a lightweight, multi-task convolutional architecture designed for real-time shoe virtual try-on (VTO) and foot pose estimation. It operates in real time across mobile browsers (WebGPU/WASM) and mobile devices, providing high-precision keypoint detection, limb affinity vector fields (PAF), and left/right foot classification.

---

## 📌 Architecture Overview

ARShoe M1v2 uses a single-stage, multi-task encoder-decoder design optimized within a strict compute budget:
- **Feature Extractor**: MobileNetV3-Small pretrained backbone extracting multi-scale representations at strides 4, 8, 16, and 32 ($64\times64$, $32\times32$, $16\times16$, $8\times8$).
- **FPN-Lite Neck**: Lateral $1\times1$ convolutions and top-down bilinear upsampling merging semantic context to stride 4 ($64\times64$, 48 channels), processed by Depthwise Separable Convolution (DSConv).
- **Decoder Heads**:
  1. **Heatmap Head**: DSConv producing `[B, 13, 64, 64]` per-keypoint Gaussian confidence heatmaps with Sigmoid activation.
  2. **PAF Head**: DSConv producing `[B, 32, 64, 64]` limb affinity vector fields (16 limb connections $\times 2$ for $x, y$ directions).
  3. **Class Head**: Dual global pooling + spatial branch producing `[B, 2, 64, 64]` pixel-wise foot classification (0: `left_foot`, 1: `right_foot`).
- **Parameter Count**: ~901,953 parameters (under 1.0M, ~69% of mobile budget).

---

## 📐 13-Keypoint Anatomical Schema

The 13-keypoint schema eliminates redundant, noisy, and ambiguous landmarks (`ball_top`, `ankle_center`, `shin_mid`), retaining the essential biomechanical foot landmarks:

```
Index 0 : toe_ground        (Inferior ground contact point under toe box)
Index 1 : heel_back         (Posterior-most prominence of calcaneus)
Index 2 : heel_ground       (Inferior ground contact point of calcaneus)
Index 3 : ball_medial       (1st metatarsal head - big-toe side)
Index 4 : ball_lateral      (5th metatarsal head - pinky-toe side)
Index 5 : instep_top        (Apex of dorsal instep)
Index 6 : arch_medial       (Apex of medial longitudinal arch)
Index 7 : midfoot_lateral   (Lateral midfoot boundary)
Index 8 : malleolus_medial  (Center of medial malleolus / inside ankle)
Index 9 : malleolus_lateral (Center of lateral malleolus / outside ankle)
Index 10: toe_tip           (Anterior tip of big toe)
Index 11: throat            (Anterior collar / shoe opening throat)
Index 12: achilles          (Achilles tendon insertion above heel)
```

### 16 Limb Connections (`configs/paf_connections_13kp.yaml`)

```
Sole Chain:
  0: toe_tip (10)        → toe_ground (0)
  1: toe_ground (0)      → heel_ground (2)
  2: heel_ground (2)     → heel_back (1)

Ball & Instep:
  3: ball_medial (3)     → ball_lateral (4)   [Metatarsal Bridge]
  4: ball_medial (3)     → instep_top (5)
  5: ball_lateral (4)    → instep_top (5)

Mid-Sole:
  6: heel_ground (2)     → arch_medial (6)
  7: heel_ground (2)     → midfoot_lateral (7)
  8: arch_medial (6)     → instep_top (5)
  9: midfoot_lateral (7) → instep_top (5)

Ankle Loop & Collar Bridge:
  10: instep_top (5)     → throat (11)
  11: throat (11)        → malleolus_medial (8)
  12: throat (11)        → malleolus_lateral (9)
  13: malleolus_medial (8) → achilles (12)
  14: malleolus_lateral (9) → achilles (12)
  15: achilles (12)      → heel_back (1)
```

---

## 🔬 Research-Grounded Supervision Fixes

The pipeline includes crucial ground-truth supervision corrections:

1. **PAF Multi-Instance Accumulation & Negative Background Supervision**:
   - Multiple foot instances accumulate unit vectors into shared tensor accumulators; contributing pixels are averaged instead of overwritten.
   - Reliable background outside feet is supervised as zero vectors (`mask = 1.0`, target = `(0, 0)`).
   - Foot bounding boxes with unlabeled endpoints ($v=0$) are masked out (`mask = 0.0`) to avoid penalizing uncertain physical limbs.
2. **Heatmap Spatial Visibility Masking**:
   - Channels where a keypoint is unlabeled on all instances still supervise background as 0.0.
   - Truly negative (empty) images supervise all channels as background.
   - Positive Gaussian targets always enforce `mask = 1.0`, ensuring overlapping bounding boxes cannot erase valid positive peaks.

---

## 🧩 Decoders Available

- **Spatial Decoder (`decode_spatial_instances_single`)**:
  - Centroid discovery with Voronoi region candidate gathering and PAF iterative refinement.
  - Production baseline: PCK@0.20 = 75.37% on validation.
- **OpenPose PAF-Led Assembly (`decode_paf_instances_single`)**:
  - Plateau-safe peak extraction with subpixel Taylor expansion.
  - PAF line integral candidate scoring along 16 limbs.
  - One-to-one disjoint bipartite matching per limb.
  - Graph tree assembly with contradiction rejection (strictly one peak per keypoint type).
  - Post-assembly left/right class assignment.
  - Achieves **0 missing feet (98.4% instance recall)** on validation.

---

## 📂 Repository Structure

```
├── shoes_vto/
│   ├── configs/
│   │   ├── paf_connections_13kp.yaml       # 16-limb PAF configuration for 13 KPs
│   │   ├── skeleton_priors_train_13kp.json # Pairwise distance priors for 13 KPs
│   │   ├── keypoint_schema_13kp.yaml       # Anatomical schema definition
│   │   ├── train_config_13kp.yaml          # Hyperparameter config
│   │   └── box_calibration.json            # Foot bounding box calibration margins
│   ├── src/
│   │   ├── models/
│   │   │   ├── arshoe_m1v2.py              # ARShoe M1v2 architecture
│   │   │   └── heads/
│   │   │       ├── heatmap_head.py         # Heatmap head + batch generator with masking
│   │   │       ├── paf_head.py             # PAF head + multi-instance accumulation generator
│   │   │       └── class_head.py           # Classification head
│   │   ├── losses/
│   │   │   ├── heatmap_loss.py             # AdaptiveWingLoss with masking
│   │   │   ├── paf_loss.py                 # PAFLoss (Smooth L1) with masking
│   │   │   └── class_loss.py               # CrossEntropy with ignore index
│   │   ├── datasets/
│   │   │   └── yolo_dataset.py             # YOLO pose dataset loader
│   │   ├── utils/
│   │   │   └── keypoint_grouping.py        # Spatial & OpenPose PAF-Led Decoders
│   │   └── training/
│   │       └── trainer_m1.py               # Differential LR trainer with live visualizations
│   ├── tests/
│   │   ├── test_paf_heatmap_supervision.py # 7 unit tests for supervision & masking
│   │   └── test_paf_assembly.py            # Unit tests for PAF-led assembly
│   ├── tools/
│   │   ├── convert_dataset_to_13kp.py      # 16-KP to 13-KP YOLO dataset converter
│   │   ├── verify_13kp_dataset.py          # Verifies 13-KP dataset integrity
│   │   ├── compute_13kp_skeleton_priors.py # Computes anatomical priors for 13-KP
│   │   ├── visualize_gt_supervision_targets.py # GT targets & masks visualizer
│   │   ├── run_fast_overfit_sanity_test_corrected.py # 8-image overfit diagnostic
│   │   ├── evaluate_decoders_on_validation.py # Full validation benchmark
│   │   └── benchmark_val_paf_vs_spatial.py # Comparative validation evaluation
│   └── train_m1v2_13kp.py                  # Main entry point for 13-KP training
└── train_13kp_colab.ipynb                  # Colab training notebook (root shortcut)
```

---

## ⚡ High-Throughput GPU Training (Colab T4 / A100)

Training speed has been optimized for NVIDIA GPUs (T4 / A100) to complete epochs in **~10 seconds** (down from ~54s):
1. **Target Grid Pre-Caching**: Global coordinate meshgrid tensors (`_GRID_CACHE_HM`, `_GRID_CACHE_PAF`) are allocated once on the GPU and reused across all steps, eliminating >37,000 redundant CPU-GPU allocator stalls per epoch.
2. **Batch Size Scaling**: `batch_size = 64` saturates GPU Tensor Cores, reducing steps per epoch from 81 to 20 while utilizing ~2.5–3.5 GB of VRAM.
3. **Optimized DataLoader**: `num_workers = 2` with `persistent_workers = True` and `pin_memory = True` perfectly matches Google Colab's 2 vCPUs without worker restart latency.
4. **Automatic Mixed Precision (AMP)**: Native FP16 training with gradient scaling for accelerated forward/backward passes.

---

## 🚀 Quickstart & Workflows

### 1. Run Unit Tests (Local)
```bash
conda activate yolo
python -m unittest shoes_vto/tests/test_paf_heatmap_supervision.py
python -m unittest shoes_vto/tests/test_paf_assembly.py
```

### 2. Run Overfit Sanity Diagnostic
```bash
python shoes_vto/tools/run_fast_overfit_sanity_test_corrected.py
```

### 3. Evaluate Decoders on Full Validation Split
```bash
python shoes_vto/tools/benchmark_val_paf_vs_spatial.py
```

### 4. Fast Retraining on Google Colab
Open [`train_13kp_colab.ipynb`](train_13kp_colab.ipynb):

```bash
# Step 1: Convert dataset to 13 keypoints (symlinks images, updates labels)
!python shoes_vto/tools/convert_dataset_to_13kp.py \
    --src /content/shoes_vto_v2/shoes_vto/data/shuffled_v4_1 \
    --dst /content/shoes_vto_v2/shoes_vto/data/shuffled_v4_1_13kp \
    --symlink

# Step 2: Launch Fast Retraining with Live Every-10-Epoch Visualizations
!python shoes_vto/train_m1v2_13kp.py \
    --dataset_root /content/shoes_vto_v2/shoes_vto/data/shuffled_v4_1_13kp \
    --output_dir shoes_vto/outputs/m1v2_13kp_stage2_paf_fix \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --num_workers 2 \
    --vis_interval 10

# Step 3: Evaluate Best Checkpoint
!python shoes_vto/tools/benchmark_val_paf_vs_spatial.py
```
