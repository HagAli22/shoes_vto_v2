# 👟 ARShoe M1v2 Training on Google Colab (Step 5 & Step 6)

This guide walks you through training the new **ARShoe M1v2** model from scratch on Google Colab using the merged `shuffled_v3 + shuffled_v4` dataset.

---

## 📦 Files Ready on Your Local PC

1. **Dataset Archive**: `dataset/merged_v3_v4.zip` (~292 MB)
   - Contains 2,034 training images, 253 validation images, and 258 test images.
   - 100% duplicate-free, verified label format, strict split separation.
2. **Source Code Archive**: `shoes_vto_code.zip` (~98 KB)
   - Contains all bug fixes (handedness-aware horizontal flip, upward leg window decoding, FPN-lite MobileNetV3 M1v2, differential LR trainer, updated evaluator).

---

## 🚀 Step-by-Step Instructions for Google Colab

### 1. Open Google Colab & Select GPU
- Go to [Google Colab](https://colab.research.google.com).
- Click **New Notebook**.
- In the top menu, go to **Runtime > Change runtime type**.
- Select **T4 GPU** (free) or **A100 GPU** (Colab Pro) and click **Save**.

---

### 2. Upload Files to Colab
In the Colab left sidebar, click the 📁 **Files** icon:
- Drag and drop `shoes_vto_code.zip` into `/content/`.
- Drag and drop `dataset/merged_v3_v4.zip` into `/content/` (or upload to Google Drive and mount drive).

---

### 3. Run Notebook Cells

#### Cell 1: Unzip Files & Install Requirements
```python
# Unzip code and dataset
!unzip -q -o /content/shoes_vto_code.zip -d /content/
!unzip -q -o /content/merged_v3_v4.zip -d /content/dataset/

# Install dependencies
!pip install -q thop albumentations opencv-python matplotlib
```

#### Cell 2: Verify Setup & GPU
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
!python -c "import sys; sys.path.insert(0, '/content/shoes_vto'); from src.models.arshoe_m1v2 import ARShoeM1v2; m = ARShoeM1v2(); print('M1v2 Params:', m.count_parameters()[0])"
```

#### Cell 3: Step 5 — Train M1v2 From Scratch (100 Epochs)
```python
%cd /content/shoes_vto

!python train_m1v2.py \
    --dataset_root /content/dataset/merged_v3_v4 \
    --output_dir outputs/m1v2_training \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0005 \
    --eta_min 0.000001 \
    --num_workers 4
```
*Note: With Colab GPU (T4/A100), 100 epochs will complete in approximately 12–18 minutes.*

#### Cell 4: Step 6 — Evaluate on Validation Split
```python
%cd /content/shoes_vto

!python evaluate_m1.py \
    --checkpoint outputs/m1v2_training/checkpoints/best.pth \
    --dataset_root /content/dataset/merged_v3_v4 \
    --split valid \
    --output_dir outputs/m1v2_valid_eval \
    --num_vis 10
```
*This will print the comprehensive PCK report (PCK@0.20, PCK@0.10, PCK@0.05, Foot-14 vs Leg-2, Left/Right class accuracy, and per-keypoint breakdown) and generate side-by-side visualizations.*

#### Cell 5: Package Checkpoints & Visualizations for Download
```python
!zip -r /content/m1v2_results.zip /content/shoes_vto/outputs/m1v2_training/checkpoints /content/shoes_vto/outputs/m1v2_valid_eval
print("Results ready for download at /content/m1v2_results.zip!")
```
*Right-click `/content/m1v2_results.zip` in the Colab file tree and click **Download**.*

---

### 4. Step 7 — Final Test Split (DO NOT RUN YET)
Per project guidelines, the test split (`/content/dataset/merged_v3_v4/test`, 258 images) remains strictly untouched until we review the validation evaluation metrics together and confirm model quality.
Once finalized, the one-time final test command will be:
```python
# ONLY RUN ONCE AFTER FINAL MODEL VALIDATION APPROVAL
!python evaluate_m1.py \
    --checkpoint outputs/m1v2_training/checkpoints/best.pth \
    --dataset_root /content/dataset/merged_v3_v4 \
    --split test \
    --output_dir outputs/m1v2_test_eval \
    --num_vis 10
```

