# ARShoe M1_V2 (14 Keypoints)

Transitioning to **M1_V2** with a refined 14-keypoint anatomical schema by removing `ankle_center` (old 12) and `shin_mid` (old 15).

---

## 1. Keypoint Schema & Indexing (14 Keypoints)

| Index | Name | Role | Symmetry Rule |
| :---: | :--- | :--- | :---: |
| **0** | `toe_ground` | Outsole toe boundary | Midline (invariant) |
| **1** | `heel_back` | Calcaneal posterior point | Midline (invariant) |
| **2** | `heel_ground` | Heel ground contact point | Midline (invariant) |
| **3** | `ball_medial` | 1st metatarsal head | Bilateral (swaps with 4) |
| **4** | `ball_lateral` | 5th metatarsal head | Bilateral (swaps with 3) |
| **5** | `ball_top` | Dorsal forefoot apex | Midline (invariant) |
| **6** | `instep_top` | Dorsal instep apex / tongue bridge | Midline (invariant) |
| **7** | `arch_medial` | Medial longitudinal arch | Bilateral (swaps with 8) |
| **8** | `midfoot_lateral`| Lateral midfoot border | Bilateral (swaps with 7) |
| **9** | `malleolus_medial` | Inner ankle bone | Bilateral (swaps with 10) |
| **10**| `malleolus_lateral`| Outer ankle bone | Bilateral (swaps with 9) |
| **11**| `toe_tip` | Distal hallux tip | Midline (invariant) |
| **12**| `throat` | Anterior shoe collar opening | Midline (invariant) |
| **13**| `achilles` | Posterior collar tendon | Midline (invariant) |

---

## 2. PAF Topology (14 Limbs = 28 Channels)

Defined in [`configs/paf_connections_14kp.yaml`](file:///c:/Users/Matrix%20Store/Downloads/Compressed/inspect_foot_KP_cocodataset/shoes_vto/configs/paf_connections_14kp.yaml):
* **Sole Chain (3):** `11→0` (toe_tip → toe_ground), `0→2` (toe_ground → heel_ground), `2→1` (heel_ground → heel_back)
* **Ball Ring (2):** `3→5` (ball_medial → ball_top), `5→4` (ball_top → ball_lateral)
* **Mid-Sole (4):** `2→7` (heel_ground → arch_medial), `2→8` (heel_ground → midfoot_lateral), `7→6` (arch_medial → instep_top), `8→6` (midfoot_lateral → instep_top)
* **Dorsal Bridge (1):** `5→12` (ball_top → throat)
* **Ankle Collar Ring (4):** `9→12` (malleolus_medial → throat), `10→12` (malleolus_lateral → throat), `9→13` (malleolus_medial → achilles), `10→13` (malleolus_lateral → achilles)

---

## 3. Architecture & Parameter Count

* **Model Class:** `ARShoeM1V2` in [`src/models/arshoe_m1_v2.py`](file:///c:/Users/Matrix%20Store/Downloads/Compressed/inspect_foot_KP_cocodataset/shoes_vto/src/models/arshoe_m1_v2.py)
* **Encoder:** Fast-SCNN V2 (24,096 params)
* **Heatmap Head:** 14 channels (74,766 params)
* **PAF Head:** 28 channels (75,676 params)
* **Class Head:** Left / Right foot classifier (22,786 params)
* **Total Parameters:** **197,324** (~15.2% of 1.3M budget)

---

## 4. Dataset

* **Dedicated 14-KP dataset:** `dataset/shuffled_v3_14kp/` (880 train, 110 valid, 112 test)
* Each annotation line contains 47 tokens (`class cx cy w h` + 14 keypoint triplets `x y v`).
* `data.yaml` has `kpt_shape: [14, 3]`.

---

## 5. How to Train

### Full Training (200 Epochs):
```bash
python train_m1_v2.py --num_epochs 200 --batch_size 16 --output_dir outputs/m1_v2_training
```

### Dry Run Test:
```bash
python train_m1_v2.py --dry_run
```

---

## 6. How to Evaluate

```bash
python evaluate_m1_v2.py --checkpoint outputs/m1_v2_training/checkpoints/best.pth --output_dir outputs/m1_v2_training --num_vis 10
```

This outputs:
1. Multi-threshold PCK: `PCK@0.20`, `PCK@0.10`, and `PCK@0.05`
2. Per-keypoint PCK table for all 14 keypoints
3. 3-panel visual comparisons saved to `outputs/m1_v2_training/visualizations/`

