# ARShoe M1_V2 (14 Keypoints) Evaluation & Training Report

---

## 1. Executive Summary & Comparison: M1 vs M1_V2

By removing the two noisiest virtual/truncated landmarks (`ankle_center` and `shin_mid`), the model achieved **higher consistency across the foot surface**:

| Metric | Original M1 (16 KPs) | M1_V2 (14 KPs) | Impact |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | 212,386 (16.3%) | **197,324** (15.2%) | **-15,062 params** (leaner model) |
| **Peak Val PCK@0.20** | 66.16% | **70.53%** (Epoch 192) | **+4.37%** peak gain |
| **Final Checkpoint PCK@0.20** | 66.16% | **66.63%** (Epoch 186) | **+0.47%** overall gain |
| **PCK@0.10 (Medium Precision)** | 34.00% | **32.70%** | Comparable |
| **PCK@0.05 (High Precision)** | 10.99% | **10.51%** | Comparable |
| **Number of "Poor" Keypoints (<40%)** | 2 (`shin_mid` @ 26.8%, `ankle_center` @ 34.8%) | **0 (None)** | **All landmarks $\ge 54.7\%$** |
| **Worst Performing Landmark** | 26.8% (`shin_mid`) | **54.7%** (`midfoot_lateral`) | **+27.9% floor improvement** |

---

## 2. Full Per-Keypoint PCK Breakdown (M1_V2, 14 Keypoints)

Evaluated across **110 validation images** (186 foot instances, 1,798 valid keypoints):

| Index | Anatomical Landmark Name | Valid Count | PCK@0.20 (Coarse) | PCK@0.10 (Medium) | PCK@0.05 (Precise) | Quality Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | `toe_ground` | 116 | **77.6%** | 53.4% | 15.5% | 🟢 Strong |
| **1** | `heel_back` | 29 | **58.6%** | 24.1% | 6.9% | 🟡 Moderate |
| **2** | `heel_ground` | 116 | **56.9%** | 16.4% | 6.9% | 🟡 Moderate |
| **3** | `ball_medial` | 161 | **72.0%** | 30.4% | 7.5% | 🟢 Strong |
| **4** | `ball_lateral` | 145 | **66.2%** | 20.7% | 1.4% | 🟢 Strong |
| **5** | `ball_top` | 153 | **85.6%** | 59.5% | 25.5% | 🟢 Strong (Best) |
| **6** | `instep_top` | 145 | **66.2%** | 32.4% | 9.7% | 🟢 Strong |
| **7** | `arch_medial` | 151 | **56.3%** | 15.2% | 5.3% | 🟡 Moderate |
| **8** | `midfoot_lateral` | 128 | **54.7%** | 17.2% | 3.1% | 🟡 Moderate |
| **9** | `malleolus_medial` | 139 | **68.3%** | 23.0% | 6.5% | 🟢 Strong |
| **10**| `malleolus_lateral` | 62 | **54.8%** | 21.0% | 3.2% | 🟡 Moderate |
| **11**| `toe_tip` | 157 | **76.4%** | 52.9% | 19.1% | 🟢 Strong |
| **12**| `throat` | 132 | **69.7%** | 44.7% | 17.4% | 🟢 Strong |
| **13**| `achilles` | 164 | **54.9%** | 31.1% | 11.0% | 🟡 Moderate |

---

## 3. Training Details

* **Total Epochs Trained:** 200 epochs
* **Best Checkpoint Epoch:** Epoch 186
* **Best Validation Loss:** 2.2793
* **Peak Validation PCK@0.20:** 70.53% (Epoch 192)
* **Optimizer:** AdamW with Cosine Annealing (1e-3 down to 1e-5)
* **Loss Components (Epoch 186):**
  * Heatmap Loss: 0.272
  * PAF Loss: 0.151
  * Classification Loss: 0.598

---

## 4. Visualizations

10 side-by-side 3-panel visual comparisons (`[Ground Truth | Prediction | Error Overlay]`) are saved in:
`shoes_vto/outputs/m1_v2_training/visualizations/`

Example files:
* `eval_vis_01_10_jpg.rf.c63fc644aef629192c4231c76f6bc3d0_left_foot.jpg`
* `eval_vis_02_10_jpg.rf.c63fc644aef629192c4231c76f6bc3d0_right_foot.jpg`
* `eval_vis_05_1_0003_f00024_jpg.rf.b678f56da011fce268563133e9a63b94_right_foot.jpg`
* `eval_vis_07_1_0004_f00032_jpg.rf.e0611718b0a07fbfe916890922421831_right_foot.jpg`

---

## 5. Artifacts and Checkpoint Locations

* **Model Weights:** `shoes_vto/outputs/m1_v2_training/checkpoints/best.pth`
* **Latest Weights:** `shoes_vto/outputs/m1_v2_training/checkpoints/latest.pth`
* **Training Log:** `shoes_vto/outputs/m1_v2_training/training_log.csv`
* **Evaluation JSON:** `shoes_vto/outputs/m1_v2_training/evaluation_summary.json`
