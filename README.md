# 🚀 ARShoe Architecture with 16 Keypoints for Browser AR

## **Implementation Plan - ARShoe Architecture with 16 Keypoints for Browser AR**

### **Problem Statement:**
Implement a shoe AR system using Paper 2's ARShoe single-stage encoder-decoder architecture, adapted with:
- 16 anatomical keypoints (from Roboflow annotated dataset)
- Explicit left/right foot classification head
- PAF-based keypoint association for robustness
- All 9 output heads from AI Plan v2
- Browser deployment target (ONNX opset 12, WebGPU/WASM)
- Production performance budget: ≤7ms WASM inference time

---

## **Requirements:**

### **Architecture:**
- Single-stage multi-task encoder-decoder (ARShoe style)
- Lightweight Fast-SCNN-inspired encoder (~1.3M parameters target)
- Multiple decoder heads sharing the backbone
- Explicit 2-class head (left_foot=0, right_foot=1)
- PAF fields for keypoint association robustness
- All outputs in single forward pass

### **Dataset:**
- Location: `dataset/shuffled_v3/` (YOLO format from Roboflow)
- Format: YOLO pose with 16 keypoints × 3 (x, y, visibility)
- Classes: 2 (left_foot, right_foot)
- Splits: train/, valid/, test/
- **CRITICAL:** Keypoint order follows `dataset/keypoint-order-audit.md`

### **16 Keypoint Schema (Actual Roboflow Order):**
```
Index 0: toe_ground (Tier 1) - Inferior contact point under toe box
Index 1: heel_back (Tier 1) - Posterior-most prominence of calcaneus
Index 2: heel_ground (Tier 1) - Inferior ground contact point of calcaneus
Index 3: ball_medial (Tier 1) - 1st metatarsal head (Medial Ball)
Index 4: ball_lateral (Tier 1) - 5th metatarsal head (Lateral Ball)
Index 5: ball_top (Tier 1) - Superior dorsal point over metatarsal heads
Index 6: instep_top (Tier 1) - Apex of the dorsal instep arch
Index 7: arch_medial (Tier 2) - Medial longitudinal arch apex
Index 8: midfoot_lateral (Tier 2) - Lateral midfoot outer boundary
Index 9: malleolus_medial (Tier 1) - Center of medial malleolus
Index 10: malleolus_lateral (Tier 1) - Center of lateral malleolus
Index 11: toe_tip (Tier 1) - Anterior tip of Big Toe (MOST IMPORTANT)
Index 12: ankle_center (Tier 1) - Talocrural joint center
Index 13: throat (Tier 1) - Anterior shoe opening throat
Index 14: achilles (Tier 1) - Achilles tendon insertion
Index 15: shin_mid (Tier 1) - Anterior lower tibia midpoint
```

### **PAF Limb Connections (15 pairs):**
```
Sole chain: 11→0, 0→2, 2→1 (toe_tip → toe_ground → heel_ground → heel_back)
Ball ring: 3→5→4 (ball_medial → ball_top → ball_lateral)
Mid-sole: 2→7, 2→8, 7→6, 8→6 (heels → arches → instep_top)
Ankle ring: 9→12→10, 12→13, 12→14 (malleoli → ankle_center → throat/achilles)
Leg axis: 12→15 (ankle → shin)
Ball-ankle: 5→13 (ball_top → throat)
```

### **Output Heads (11 total):**
1. **Heatmaps** `[B, 16, H/4, W/4]` - Per-keypoint confidence maps
2. **PAFs** `[B, 30, H/4, W/4]` - 15 limb connections × 2 (x,y vectors)
3. **Class** `[B, 2, H/4, W/4]` - Left/right foot classification
4. **Keypoints** `[B, 16, 3]` - Final (x, y, visibility)
5. **Rotation** `[B, 6]` - 6-D continuous rotation
6. **Mask** `[B, 2, 160, 160]` - foot/shoe + occluder channels
7. **Sigma** `[B, 16]` - Per-keypoint uncertainty
8. **Confidence** `[B, 16]` - Per-keypoint confidence (smoothstep of sigma)
9. **Presence** `[B, 1]` - Foot present in frame
10. **Scale** `[B, 2]` - foot_length, foot_width
11. **Scale_sigma** `[B, 2]` - Scale uncertainty

### **Performance Targets:**
- Model size: ≤14 MB fp32, ≤7 MB fp16, ≤4 MB int8
- Inference: ≤7 ms WASM per foot (≤16 ms for 2 feet)
- Accuracy: mAP@50 ≥ 0.80, rotation error ≤6° median, mask IoU ≥0.90

### **Deployment:**
- ONNX opset 12 export
- WebGPU and WASM EP compatibility
- Static shapes (except batch dimension)
- Sigmoid/softmax baked in-graph

### **Python Environment:**
- Use: `C:\Users\miniconda3\envs\yolo\python.exe` (all required libraries already installed)

---

## **Network Architecture:**

```
Input: [1, 3, 256, 256] RGB fp32 [0,1]
                │
                ▼
    ┌───────────────────────────┐
    │  Fast-SCNN Encoder        │
    │  - Initial Conv (stride 2) │
    │  - 3× DSConv blocks       │
    │  - Feature Extractor      │
    │  Output: [1, 128, 64, 64] │
    └──────────┬────────────────┘
               │ Shared Features
               ├──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
               ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
           ┌─────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
           │Heat │ │PAF │ │Cls │ │KPs │ │Rot │ │Mask│ │Sig │ │Conf│ │Pres│ │Scal│ │S_σ │
           │ Map │ │    │ │    │ │    │ │    │ │    │ │    │ │    │ │    │ │    │ │    │
           └─────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘
             16ch   30ch   2ch    16×3   6ch   2×160² 16ch   16ch   1ch    2ch    2ch
            64×64  64×64  64×64  global global 160×160 global global global global global
```

---

## **Task Progress:**

### ✅ **Task 1: Project Setup and Dataset Conversion** - **DONE**
**Objective:** Set up project structure, verify dataset, and create YOLO→internal format converter

**Status:** ✅ COMPLETED
**Completed on:** 2026-09-26

**What was done:**
- ✅ Created project directory structure following AI Plan v2 layout
- ✅ Verified dataset at `dataset/shuffled_v3/` (train/valid/test splits)
- ✅ Read and documented keypoint order from `keypoint-order-audit.md`
- ✅ Created configuration files (keypoint_schema.yaml, paf_connections.yaml, train_config.yaml)
- ✅ Implemented YOLO dataset loader (`yolo_dataset.py`)
- ✅ Created visualization utilities (`visualizer.py`)
- ✅ Created foot_canonical.json with 16-KP reference (265mm foot)
- ✅ Verified Python environment using `C:\Users\miniconda3\envs\yolo\python.exe`
- ✅ Created test script (`test_task1.py`) - all tests passed

**Test Results:**
- ✅ All required directories created
- ✅ All configuration files are valid YAML
- ✅ foot_canonical.json validated (16 keypoints, Index 11 = toe_tip)
- ✅ Dataset loader successfully loads train/valid/test splits
- ✅ Sample loaded correctly: torch.Size([3, 256, 256])
- ✅ First instance has correct class and 16 keypoints
- ✅ toe_tip (Index 11) verified at correct position
- ✅ All required Python packages installed

**Dataset Statistics:**
```
TRAIN:   880 images, 1488 instances (752 left, 736 right), 64.7% keypoint visibility
VALID:   110 images,  186 instances (100 left,  86 right), 65.7% keypoint visibility
TEST:    112 images,  187 instances ( 92 left,  95 right), 66.8% keypoint visibility
TOTAL:  1102 images, 1861 instances, 29,776 keypoints (19,357 visible)
```

**Key Files Created:**
- `shoes_vto/configs/keypoint_schema.yaml` - 16-KP schema from audit
- `shoes_vto/configs/paf_connections.yaml` - 15 limb pairs for skeleton
- `shoes_vto/configs/train_config.yaml` - Training hyperparameters
- `shoes_vto/deliverables/foot_canonical.json` - Reference 3D foot model
- `shoes_vto/src/dataset/yolo_dataset.py` - YOLO format data loader
- `shoes_vto/src/dataset/visualizer.py` - Visualization utilities
- `shoes_vto/test_task1.py` - Verification test (all passed ✅)

---

### ✅ **Task 2: Implement Fast-SCNN Encoder Backbone** - **DONE**
**Objective:** Build lightweight encoder following ARShoe's Fast-SCNN design

**Status:** ✅ COMPLETED
**Completed on:** 2026-09-26

**What was done:**
- ✅ Implemented `DepthwiseSeparableConv` block (depthwise + pointwise convolutions)
- ✅ Implemented `DSConvBlock` wrapper with configurable stride
- ✅ Created `FastSCNNEncoder` V1 (4-stage with upsampling, 60K params)
- ✅ Created `FastSCNNEncoderV2` V2 (more efficient, 24K params) - **RECOMMENDED**
- ✅ Added `create_encoder()` factory function
- ✅ Parameter counting utilities (count + breakdown)
- ✅ Created comprehensive test suite (`test_task2.py`) - ALL TESTS PASSED ✅

**Test Results:**
- ✅ Encoder V1: [1, 3, 256, 256] → [1, 128, 64, 64], 60,192 params
- ✅ Encoder V2: [1, 3, 256, 256] → [1, 128, 64, 64], 24,096 params
- ✅ Output stride: 4 (64×64 feature maps for 256×256 input)
- ✅ Gradient flow verified (backward pass works)
- ✅ Batch flexibility tested (1, 2, 4, 8, 16)
- ✅ Eval mode works (ONNX export ready)
- ✅ No NaN/Inf in outputs
- ✅ Parameter budget: 24,096 / 1,300,000 (1.85% of encoder budget!)

**Architecture (V2 - Recommended):**
```
Input [1, 3, 256, 256]
  → Initial Conv: 3→32 ch, stride=2 (128×128)
  → DSConv1: 32→64 ch, stride=2 (64×64)
  → DSConv2: 64→96 ch, stride=1 (64×64)
  → DSConv3: 96→128 ch, stride=1 (64×64)
Output [1, 128, 64, 64]
```

**Key Achievement:**
- Ultra-efficient: Only 24K params leaves 1.28M params for decoder heads
- V2 is 60% more efficient than V1 (no upsampling, simpler architecture)
- Ready to share features across all 11 decoder heads

**Files Created:**
- `shoes_vto/src/models/encoder.py` (370+ lines, 2 encoder versions)
- `shoes_vto/test_task2.py` (250+ lines, 9 comprehensive tests)
- `shoes_vto/outputs/TASK2_COMPLETION_REPORT.md` (detailed report)

---

### ✅ **Task 3: Implement Heatmap and PAF Decoder Heads** - **DONE**
**Objective:** Build heatmap and PAF decoders for 16-KP detection with limb associations

**Status:** ✅ COMPLETED
**Completed on:** 2026-09-26

**What was done:**
- ✅ Implemented `HeatmapHead` (16 channels @ 64×64 with sigmoid)
- ✅ Implemented `HeatmapHeadWithPixelShuffle` (alternative with PixelShuffle)
- ✅ Implemented `PAFHead` (30 channels @ 64×64 for 15 limbs × 2)
- ✅ Created `generate_gaussian_heatmap()` for GT generation
- ✅ Created `generate_heatmaps_batch()` for batch GT
- ✅ Created `generate_paf_field()` for single limb PAF
- ✅ Created `generate_pafs_batch()` for batch PAF GT
- ✅ Implemented `HeatmapLoss` (MSE with masking)
- ✅ Implemented `FocalHeatmapLoss` (focal loss variant)
- ✅ Implemented `AdaptiveWingLoss` (robust alternative)
- ✅ Implemented `PAFLoss` (Smooth L1 with masking)
- ✅ Implemented `PAFMSELoss` (MSE alternative)
- ✅ Implemented `PAFVectorAngleLoss` (angle-aware loss)
- ✅ Created comprehensive test suite (`test_task3.py`) - ALL 11 TESTS PASSED ✅

**Test Results:**
- ✅ HeatmapHead: [4, 128, 64, 64] → [4, 16, 64, 64], range [0, 1]
- ✅ PAFHead: [4, 128, 64, 64] → [4, 30, 64, 64], 75,806 params
- ✅ Gaussian peaks correctly centered at specified coordinates
- ✅ Batch GT generation works with real keypoint data
- ✅ PAF config loaded: 15 limb pairs verified
- ✅ PAF vector fields generated with correct unit vectors
- ✅ HeatmapLoss computes correctly with masking
- ✅ PAFLoss computes correctly with mask expansion (15→30 channels)
- ✅ Integration test: heads + losses work together
- ✅ Gradient flow verified (backward pass works)

**Parameter Count:**
- HeatmapHead: 74,896 params
- PAFHead: 75,806 params
- **Total heads so far:** 150,702 params
- **With encoder (24K):** 174,798 params
- **Budget remaining:** 1,125,202 / 1,300,000 (86.5%)

**Ground Truth Generation:**
- Gaussian heatmaps with σ=2 pixels on 64×64 map
- PAF vector fields with 8-pixel width along limbs
- Automatic masking for unlabeled keypoints/limbs
- Supports multiple instances per image (max aggregation)

**Loss Functions:**
- HeatmapLoss: MSE with per-keypoint masking (RECOMMENDED for M1)
- PAFLoss: Smooth L1 with per-limb masking (RECOMMENDED for M1)
- Alternative losses available for fine-tuning later

**Files Created:**
- `shoes_vto/src/models/heads/heatmap_head.py` (270+ lines)
- `shoes_vto/src/models/heads/paf_head.py` (270+ lines)
- `shoes_vto/src/losses/heatmap_loss.py` (220+ lines, 3 loss variants)
- `shoes_vto/src/losses/paf_loss.py` (240+ lines, 3 loss variants)
- `shoes_vto/test_task3.py` (340+ lines, 11 comprehensive tests)

---

### ✅ **Task 4: Implement Class Head and Left/Right Classification** - **DONE**
**Objective:** Add explicit left/right foot classification head

**Status:** ✅ COMPLETED
**Completed on:** 2026-09-26

**What was done:**
- ✅ Implemented `ClassHead` (2 channels @ 64×64 with softmax)
- ✅ Created `generate_class_map()` for single instance GT
- ✅ Created `generate_class_maps_batch()` for batch GT with ignore label
- ✅ Created `extract_class_from_keypoints()` for post-processing
- ✅ Implemented `ClassLoss` (Cross-Entropy with ignore_index)
- ✅ Implemented `FocalClassLoss` (focal loss for imbalance)
- ✅ Implemented `DiceLoss` (dice loss for segmentation)
- ✅ Created `compute_class_accuracy()` for evaluation
- ✅ Created comprehensive test suite (`test_task4.py`) - ALL 10 TESTS PASSED ✅

**Test Results:**
- ✅ ClassHead: [4, 128, 64, 64] → [4, 2, 64, 64], softmax sum = 1.0
- ✅ Forward logits match forward probs after softmax
- ✅ Class map generation with bbox and ignore regions
- ✅ Batch GT generation handles multiple instances per image
- ✅ Class extraction from keypoints works correctly
- ✅ ClassLoss computes correctly with ignore_index=-1
- ✅ Perfect predictions have 100% accuracy
- ✅ Integration test: head + loss + gradients work together
- ✅ All 3 loss variants functional

**Parameter Count:**
- ClassHead: 36,994 params (optimized with 32 intermediate channels)
- **Total model so far:** 211,792 params
  - Encoder: 24,096
  - HeatmapHead: 74,896
  - PAFHead: 75,806
  - ClassHead: 36,994
- **Budget remaining:** 1,088,208 / 1,300,000 (83.7%)
- **Budget used:** 16.3%

**Ground Truth Generation:**
- Class maps based on bbox regions
- Ignore label (-1) for background/ambiguous regions
- Supports overlapping instances (later instance overwrites)
- Automatic masking for unlabeled pixels

**Loss Function:**
- ClassLoss: Cross-Entropy with ignore_index (RECOMMENDED for M1)
- Handles -1 ignore label automatically
- Per-pixel classification loss
- Optional class weights for imbalance

**Critical Feature - Class Flip Prevention:**
- Explicit 2-class head prevents catastrophic class flips
- Target: ≤1 flip per 500 frames (hard gate)
- Per-pixel classification provides spatial consistency
- Can be integrated with PAF grouping for robustness

**Files Created:**
- `shoes_vto/src/models/heads/class_head.py` (340+ lines)
- `shoes_vto/src/losses/class_loss.py` (260+ lines, 3 loss variants)
- `shoes_vto/test_task4.py` (280+ lines, 10 comprehensive tests)

---

### ✅ **Task 5: Implement Keypoint Grouping with PAF and Class Integration** - **DONE**
**Objective:** Group heatmap peaks into coherent instances using PAF and class information

**Status:** ✅ COMPLETE

**Deliverables:**
- `src/utils/keypoint_grouping.py` (418 lines)
  - `detect_peaks()` - NMS-based peak detection
  - `detect_all_peaks()` - Batch peak detection for all keypoints
  - `compute_paf_score()` - PAF vector alignment scoring
  - `find_connections()` - PAF-based connection validation
  - `greedy_assembly()` - Instance assembly with merging
  - `assign_class_to_instances()` - Left/right classification voting
  - `group_keypoints()` - Complete pipeline (heatmaps + PAFs + class → instances)
- `test_task5.py` - 10 comprehensive tests (all passing ✅)
- `deliverables/TASK5_COMPLETION_REPORT.md`

**Key Features:**
- Config-driven PAF connections (loads from `paf_connections.yaml`)
- Greedy algorithm optimized for ≤2 feet per frame
- Class voting across all visible keypoints
- Structured output: `[{keypoints, score, num_keypoints, class_id, class_confidence}, ...]`

**Test Results:** All 10/10 tests passed ✅

---

### ✅ **Task 6: Training Pipeline M1 - Basic 16-KP Detection** - **DONE**
**Objective:** Train full detection pipeline (Encoder + Heatmap + PAF + Class heads)

**Status:** ✅ COMPLETE

**Deliverables:**
- `src/models/arshoe_m1.py` - Complete M1 model class (211,792 params)
- `src/training/trainer_m1.py` - Training loop with multi-task loss
- `train_m1.py` - Main training script
- `test_task6.py` - 7 comprehensive tests (all passing ✅)
- `test_training_setup.py` - Detailed setup verification

**Components:**
- **ARShoeM1 Model:** Integrates encoder + 3 decoder heads
- **Trainer:** Multi-task training with weighted losses, validation, checkpointing
- **Data Pipeline:** YOLO dataset loading with custom collate function
- **Loss Integration:** Heatmap (MSE), PAF (Smooth L1), Class (Cross-Entropy)

**Configuration:**
- Batch size: 16
- Learning rate: 0.001 (Adam optimizer)
- Loss weights: HM=4.0, PAF=1.0, Class=2.0
- Epochs: 50 (for initial M1 training)
- Device: CPU/CUDA auto-detect

**Test Results:** All 7/7 tests passed ✅
1. Model creation (211K params)
2. Forward pass (correct shapes/ranges)
3. Dataset loading (880 train, 110 val)
4. DataLoader with collate_fn
5. GT generation (heatmaps, PAFs, class maps)
6. Loss computation (all losses valid)
7. Backward pass (gradients flowing)

**Ready for Training:** `python train_m1.py`

---

### ⬜ **Task 7: Implement Rotation Head (6-D Continuous)** - **TODO**
**Objective:** Add rotation estimation head for 6DoF pose

**Status:** ⬜ NOT STARTED

---

### ⬜ **Task 8: Implement Mask Head (2-Channel Segmentation)** - **TODO**
**Objective:** Add segmentation mask for foot/shoe and occluders

**Status:** ⬜ NOT STARTED

---

### ⬜ **Task 9: Implement Uncertainty Heads (Sigma, Confidence)** - **TODO**
**Objective:** Add per-keypoint uncertainty estimation

**Status:** ⬜ NOT STARTED

---

### ⬜ **Task 10: Implement Presence and Scale Heads** - **TODO**
**Objective:** Add foot presence detection and size estimation

**Status:** ⬜ NOT STARTED

---

### ⬜ **Task 11: Training Pipeline M2-M4 - Full Multi-Task Model** - **TODO**
**Objective:** Train complete model with all 11 decoder heads using progressive strategy

**Status:** ⬜ NOT STARTED

---

### ⬜ **Task 12: ONNX Export with Opset 12 Compatibility** - **TODO**
**Objective:** Export trained model to ONNX for browser deployment

**Status:** ⬜ NOT STARTED

---

### ⬜ **Task 13: Post-Processing and JavaScript Integration** - **TODO**
**Objective:** Implement post-processing pipeline in JavaScript for browser deployment

**Status:** ⬜ NOT STARTED

---

### ⬜ **Task 14: Validation and Performance Optimization** - **TODO**
**Objective:** Comprehensive evaluation and optimization to meet production targets

**Status:** ⬜ NOT STARTED

---

### ⬜ **Task 15: Final Deliverables and Documentation** - **TODO**
**Objective:** Package all outputs and create comprehensive documentation

**Status:** ⬜ NOT STARTED

---

## **Milestone Summary:**

| Milestone | Tasks | Target Metrics | Status |
|-----------|-------|----------------|--------|
| **M1: Basic Detection** | Tasks 1-6 | mAP@50 ≥ 0.70, Class Acc ≥ 0.95 | 🔄 IN PROGRESS (4/6 done) |
| **M2: Add Rotation** | Task 7 | Rotation error ≤15° median | ⬜ TODO |
| **M3: Add Mask** | Task 8 | Mask IoU ≥ 0.85 | ⬜ TODO |
| **M4: Add Uncertainty** | Tasks 9-11 | All targets met | ⬜ TODO |
| **Deployment** | Tasks 12-15 | WASM ≤7ms, All deliverables | ⬜ TODO |

**Overall Progress: 4/15 tasks complete (26.7%)**

---

## **Completed Tasks Detail:**

### **✅ Task 1 Summary**
- **Completion Date:** 2026-09-26
- **Total Files Created:** 7 configuration/code files
- **Dataset Verified:** 1102 images, 1861 foot instances, 16 keypoints each
- **Critical Finding:** Index 11 = toe_tip (verified from keypoint-order-audit.md)
- **All Tests:** PASSED ✅

### **✅ Task 4 Summary**
- **Completion Date:** 2026-09-26
- **Total Files Created:** 3 code files (1 head + 1 loss + 1 test)
- **Class Head:** 36,994 params, outputs [B, 2, 64, 64]
- **Key Achievement:** Explicit left/right classification prevents catastrophic class flips
- **All Tests:** PASSED ✅ (10/10 tests)
- **Total Model:** 211,792 params (16.3% of budget)

---

## **Critical Constraints:**

1. **Python Environment:** MUST use `C:\Users\miniconda3\envs\yolo\python.exe`
2. **Dataset:** YOLO format at `dataset/shuffled_v3/` (DO NOT modify)
3. **Keypoint Order:** Follow `dataset/keypoint-order-audit.md` (Index 11 = toe_tip)
4. **ONNX Opset:** Must be 12 (browser compatibility)
5. **Performance:** ≤7ms WASM inference (non-negotiable)
6. **Model Size:** ≤14MB fp32, ≤7MB fp16, ≤4MB int8

---

## **Next Steps:**
1. ✅ Task 1: Project setup and dataset verification - **DONE**
2. ✅ Task 2: Implement Fast-SCNN encoder - **DONE**
3. ✅ Task 3: Implement heatmap and PAF heads - **DONE**
4. ✅ Task 4: Implement class head - **DONE**
5. ✅ Task 5: Implement keypoint grouping - **DONE**
6. ✅ Task 6: Build training pipeline M1 - **DONE**
7. ⬜ Task 7: Implement rotation head - **NEXT** (or continue with remaining tasks)
8. Update this README after completing each task

**Current Focus:** Training pipeline M1 complete, ready for model training or continue with Task 7+

---

**Document Version:** 1.6
**Last Updated:** 2026-09-26 (Task 6 Complete)  
**Status:** Tasks 1-6 Complete ✅ → Ready for Training or Task 7+
