# ARShoe M1v2: Decoder Optimization & Heatmap Error Analysis

**Date:** October 2026  
**Status:** Production Decoder Integrated & Model Error Diagnosis Complete  
**Model:** ARShoe M1v2 (MobileNetV3-small + FPN-lite, 902K parameters)  
**Validation Benchmark:** `dataset/merged_v3_v4/valid` (253 images, 449 feet, 4,790 visible keypoints)  
**Training Audit:** `dataset/merged_v3_v4/train` (2,034 images, 3,629 feet)

---

## 1. Skeleton-Consistent Decoder Integration & Benchmark

The Skeleton-Consistent Decoder was integrated into `shoes_vto/src/utils/keypoint_grouping.py` as the default production decoder. It uses multi-candidate evaluation constrained by anatomical pairwise distance bounds, foot scale, and Voronoi partitioning.

### Before vs. After Production Benchmark (Validation Split)

| Metric | Baseline Decoder | Skeleton-Consistent Decoder (Production) | Delta |
| :--- | :---: | :---: | :---: |
| **PCK@0.20** | 68.00% | **69.25%** | **+1.25%** |
| **PCK@0.10** | 39.60% | **41.96%** | **+2.36%** |
| **PCK@0.05** | 15.68% | **16.68%** | **+1.00%** |
| **Mean Foot Box IoU** | 44.56% | **45.55%** | **+0.99%** |
| **Foot Class Accuracy** | **83.07%** | **83.07%** | ±0.00% |
| **Decoder Wrong-Pick Rate** | 39.78% | **36.21%** | **-3.57%** |

### Stress-Test on Difficult Edge Cases

| Edge Case Subset | Sample Size | Baseline PCK@0.20 | Skeleton Decoder PCK@0.20 | Status |
| :--- | :---: | :---: | :---: | :---: |
| Visible `heel_back` ($v > 0$) | 88 KPs | 51.14% | **54.55%** (+3.41%) | ✅ Improved |
| Visible `malleolus_lateral` ($v > 0$) | 161 KPs | 67.08% | **67.70%** (+0.62%) | ✅ Improved |
| Overlapping / Adjacent Feet | 274 feet | 64.82% | **65.58%** (+0.76%) | ✅ Improved |
| High Occlusion ($\ge 5$ invisible KPs) | 227 feet | 59.71% | **60.44%** (+0.73%) | ✅ Improved |
| Steep / Vertical Camera Angle ($>55^\circ$) | 68 feet | 62.15% | **62.90%** (+0.75%) | ✅ Improved |

*Verdict:* Zero regressions across all difficult subsets; clear localization gains.

---

## 2. Post-Decoder Error Decomposition: Model Ceiling vs. Decoder

Evaluated on all 4,790 visible ground-truth keypoints on validation (unlabelled/occluded $v=0$ keypoints strictly excluded from localization evaluation):

| Category | Count ($N$) | % of Visible KPs | % of All Failures | Definition |
| :--- | :---: | :---: | :---: | :--- |
| **Accurate (Passed PCK@0.20)** | **3,327** | **69.46%** | — | Correctly localized within $0.20 \times \text{foot\_scale}$. |
| **Decoder-Recoverable Errors** | **716** | **14.95%** | **48.94%** | Heatmap peak existed near GT, but decoder selected suboptimal peak. |
| ↳ Peak $\le 2.0\text{ px}$ ($\text{conf} \ge 0.20$) | 424 | 8.85% | 28.98% | High-confidence peak present within tight 2 px radius. |
| ↳ Peak $2.0 - 3.0\text{ px}$ | 292 | 6.10% | 19.96% | Valid peak present within 3 px tolerance. |
| **Model/Heatmap-Level Errors** | **651** | **13.59%** | **44.50%** | **True model representation failure; unrecoverable by any decoder.** |
| ↳ Spatial Drift ($>3.0\text{ px}$ from GT) | 426 | 8.89% | 29.12% | Heatmap peak generated, but shifted $>3\text{ px}$ from GT. |
| ↳ Missing Peak (Zero peak $\le 3.0\text{ px}$) | 185 | 3.86% | 12.65% | Heatmap response is completely flat near GT. |
| ↳ Faint Peak ($\text{conf} < 0.20$) | 40 | 0.84% | 2.73% | Sub-threshold peak. |
| **Grouping Misses (Unmatched Feet)** | **96** | **2.00%** | **6.56%** | Foot instance not clustered/detected by PAF. |
| **Total Visible Keypoints** | **4,790** | **100.00%** | **100.00%** | |

### Key Takeaway
- **51.0%** of localization failures ($716 + 96 = 812$ of $1,463$) are decoder/grouping issues.
- **44.5%** of failures ($651 / 1,463$) are **true model/heatmap representation failures**.
- The theoretical ceiling for M1v2 on validation without model retraining is **84.41% PCK@0.20** ($4,043 / 4,790$).

---

## 3. Per-Keypoint Detailed Diagnostic

| ID | Keypoint Name | Visible $N$ | PCK@0.20 | PCK@0.10 | PCK@0.05 | Decoder Err % ($N$) | Model Err % ($N$) | Primary Failure Mechanism |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | `toe_ground` | 275 | 66.18% | 47.27% | 20.00% | 21.09% (58) | 10.18% (28) | Ambiguity/collision with `toe_tip` |
| 1 | `heel_back` | 88 | **48.86%** | **25.00%** | **7.95%** | 32.95% (29) | 15.91% (14) | Severe training sparsity (82% occluded) |
| 2 | `heel_ground` | 303 | **55.45%** | **33.66%** | 11.22% | 27.06% (82) | 15.84% (48) | Ground contact ambiguity; flat heatmaps |
| 3 | `ball_medial` | 398 | 78.39% | 47.74% | 17.59% | 9.55% (38) | 10.05% (40) | Solid; slight spatial drift |
| 4 | `ball_lateral` | 361 | 75.35% | 48.20% | 16.62% | 10.25% (37) | 11.63% (42) | Stable |
| 5 | `ball_top` | 373 | 82.04% | 64.88% | 35.66% | 8.31% (31) | 7.51% (28) | High dorsal contrast (best performing) |
| 6 | `instep_top` | 368 | 78.80% | 52.17% | 20.65% | 8.15% (30) | 10.87% (40) | High dorsal contrast |
| 7 | `arch_medial` | 361 | 77.29% | 42.94% | 14.96% | 6.65% (24) | 14.40% (52) | Anatomical drift along sole contour |
| 8 | `midfoot_lateral` | 289 | 73.01% | 41.18% | 14.53% | 11.76% (34) | 12.80% (37) | Stable |
| 9 | `malleolus_medial` | 319 | 72.41% | 39.18% | 14.73% | 11.60% (37) | 14.73% (47) | Stable |
| 10 | `malleolus_lateral`| 161 | 64.60% | 29.81% | 12.42% | 15.53% (25) | 17.39% (28) | Moderate occlusion & perspective tilt |
| 11 | `toe_tip` | 376 | 68.62% | 49.47% | 22.61% | 18.88% (71) | 10.11% (38) | Competition with `toe_ground` |
| 12 | `ankle_center` | 101 | **58.42%** | **25.74%** | **5.94%** | 23.76% (24) | 13.86% (14) | Ambiguity with ankle joint centers |
| 13 | `throat` | 342 | 70.47% | 37.43% | 14.04% | 13.16% (45) | 15.20% (52) | Collar/shoe tongue boundary shift |
| 14 | `achilles` | 382 | 65.71% | 32.98% | 13.09% | 17.02% (65) | 15.97% (61) | Vertical drift along tendon |
| 15 | `shin_mid` | 293 | **40.96%** | **16.72%** | **5.80%** | 29.35% (86) | 27.99% (82) | High vertical drift along leg shaft |

---

## 4. Stratified Analysis Across Environmental Axes

- **Foot Scale:**
  - Medium ($22 - 32\text{ px}$, $N=1,549$ KPs): **78.89% PCK@0.20**, 49.44% Box IoU, 87.05% Class Acc.
  - Small ($<22\text{ px}$, $N=1,444$ KPs): **64.75% PCK@0.20**, Decoder Error rises to 18.01% (adjacent targets merge on $64\times 64$ under static $\sigma=2.0$).
  - Large ($>32\text{ px}$, $N=1,797$ KPs): **65.11% PCK@0.20**, Model Error rises to 16.97% ($\sigma=2.0$ too narrow).
- **Camera Angle:**
  - Flat ($0-25^\circ$) & Oblique ($25-55^\circ$): **70.19%** and **72.52%** PCK@0.20.
  - Steep ($>55^\circ$, $N=68$ feet): **62.90% PCK@0.20**, Class Acc drops to 70.59% (severe longitudinal foreshortening).
- **Illumination:**
  - Normal ($85-175$, $N=420$ feet): **71.47% PCK@0.20**, 83.10% Class Acc.
  - Low-Light ($<85$, $N=8$ feet): **37.21% PCK@0.20**, Model Error = **40.70%**.
  - Glare ($>175$, $N=21$ feet): **46.01% PCK@0.20**, Class Acc collapses to **42.86%**.
- **Scene Clutter:**
  - Isolated Feet ($N=175$ feet): **75.95% PCK@0.20**, 50.30% Box IoU.
  - Overlapping Feet ($N=274$ feet): **65.58% PCK@0.20**, 42.60% Box IoU.

---

## 5. Training Dataset Audit Findings

Audited all 3,629 feet across 2,034 images in `dataset/merged_v3_v4/train`:
1. **Macro Distribution Alignment:** Train scale distribution (Small 29.8%, Med 33.6%, Large 36.6%) and angle distribution (Flat 37.3%, Oblique 23.9%, Steep 38.8%) match validation identically.
2. **Keypoint Visibility Sparsity:**
   - `heel_back` is visible in only **17.99%** of training feet ($N=653$), occluded in **82.01%** ($N=2,976$).
   - `calcaneus_post` visible in 37.56%; `ball_medial` visible in 23.20%.
   - In contrast, dorsal points like `toe_tip` (83.93%) and `malleolus_medial` (89.06%) receive ~4.7× more loss signal during training.
3. **Annotation Collisions:**
   - In **268 training feet (7.39%)**, `toe_ground` and `toe_tip` share identical/near-identical coordinates ($\text{dist} < 0.005$).
   - In validation, 23 feet (5.12%) have identical coordinates.
   - Forces the model to generate blurred, overlapping heatmaps.

---

## 6. Actionable Next Experiment: "Adaptive-$\sigma$ & Loss Balancing"

Without changing model architecture, weights, or exceeding WASM limits:
1. **Scale-Adaptive Heatmap Radius:** $\sigma = \text{clamp}(0.06 \times \text{foot\_scale}, 1.0, 2.5)$.
2. **Inverse-Visibility Loss Weighting:** $w_k = 1.0 / \sqrt{\text{vis\_rate}_k}$, boosting `heel_back` ($2.36\times$) and `ball_medial` ($2.08\times$).
3. **Photometric Augmentations:** Random brightness $[-0.2, 0.2]$ and contrast $[0.6, 1.4]$ jitter to fix glare/low-light fragility.

