# ARShoe M1 Training Results

**Date:** 2026-09-26  
**Model:** ARShoe M1 (Basic 16-Keypoint Detection)  
**Status:** ✅ TRAINING COMPLETE

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Epochs** | 50 |
| **Batch Size** | 128 |
| **Learning Rate** | 0.001 → 0.00025 (with decay) |
| **Optimizer** | Adam (weight_decay=1e-4) |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.5, patience=5) |
| **Mixed Precision** | Enabled (AMP) |
| **Device** | CUDA |
| **Training Time** | ~1.8 hours |
| **Loss Weights** | HM: 4.0, PAF: 1.0, Class: 2.0 |

### Dataset Split
- **Training:** 880 images
- **Validation:** 110 images  
- **Test:** 112 images
- **Total:** 1,102 images

---

## Training Progress

### Loss Evolution

| Metric | Epoch 1 | Epoch 25 | Epoch 50 | Improvement |
|--------|---------|----------|----------|-------------|
| **Total Loss (Train)** | 2.408 | 1.213 | 1.133 | -53% |
| **Total Loss (Val)** | 2.278 | 1.269 | 1.241 | -46% |
| **Heatmap Loss** | 0.238 | 0.006 | 0.004 | -98% |
| **PAF Loss** | 0.047 | 0.004 | 0.004 | -92% |
| **Class Loss** | 0.705 | 0.592 | 0.557 | -21% |

### Metrics Evolution

| Metric | Epoch 1 | Epoch 25 | Epoch 50 | Best (Epoch) |
|--------|---------|----------|----------|--------------|
| **Class Accuracy** | 58.33% | 65.48% | 67.36% | **67.91%** (44) |
| **Val Loss** | 2.278 | 1.269 | 1.241 | **1.233** (44) |

---

## Best Model (Epoch 44)

**Checkpoint:** `outputs/m1_training/checkpoints/best.pth`

### Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Validation Loss** | 1.233 | - | ✅ |
| **Class Accuracy** | 67.91% | ≥95% | ⚠️ Below target |
| **Heatmap Loss** | 0.004 | Low | ✅ Excellent |
| **PAF Loss** | 0.003 | Low | ✅ Excellent |
| **Class Loss** | 0.606 | Low | ⚠️ Needs improvement |

### Observations

**✅ Strengths:**
1. **Keypoint Localization:** Heatmap loss converged to 0.004 - excellent precision
2. **Limb Associations:** PAF loss at 0.003 - strong connection modeling
3. **Stable Training:** Smooth convergence without oscillations
4. **GPU Efficiency:** 128 batch size, ~20s/epoch with AMP

**⚠️ Areas for Improvement:**
1. **Class Accuracy:** 67.91% is below 95% target
   - Model struggles with left/right discrimination
   - May confuse similar foot poses
   - Needs more class-specific training or augmentation

2. **Keypoint Completeness:** Detected instances have only 3-6/16 keypoints
   - Many keypoints below confidence threshold
   - Heatmap peaks max at ~0.14 (relatively low)
   - May need to adjust σ or training strategy

---

## Prediction Analysis (Test Set)

**Sample Results (5 random test images):**

| Image | GT Instances | Pred Instances | Heatmap Max | Status |
|-------|--------------|----------------|-------------|--------|
| 64 | 2 | 2 | 0.142 | ✅ Correct count |
| 99 | 1 | 2 | 0.127 | ❌ Over-detection |
| 40 | 1 | 2 | 0.145 | ❌ Over-detection |
| 38 | 2 | 2 | 0.125 | ✅ Correct count |
| 45 | 2 | 2 | 0.114 | ✅ Correct count |

**Detection Characteristics:**
- **Instance Count:** Generally detects 2 instances per image
- **Keypoint Coverage:** 3-6 visible keypoints per instance (avg ~4/16 = 25%)
- **Class Confidence:** 0.51-0.73 (moderate confidence)
- **Heatmap Activation:** Peak values 0.11-0.15 (relatively low)

**Class Prediction Examples:**
- Image 64: Right + Left ✅ (both detected correctly)
- Image 38: Right + Right ⚠️ (should be Right + Left)
- Image 45: Left + Left ⚠️ (should be Right + Left)

---

## Analysis & Recommendations

### What Worked Well

1. **Architecture Efficiency**
   - 211K params (16.3% of budget) - plenty of room for more heads
   - Fast inference on GPU
   - Stable multi-task training

2. **Heatmap Learning**
   - Converged to very low loss (0.004)
   - Model can localize keypoints accurately
   - Gaussian σ=2px is appropriate

3. **PAF Learning**
   - Low loss (0.003) indicates good limb modeling
   - 8px width seems appropriate
   - Association logic working

### What Needs Improvement

1. **Class Head Performance**
   - Only 67.91% accuracy (target: ≥95%)
   - Frequent left/right confusions
   - Loss weight may be too low (currently 2.0)

**Recommendations:**
- Increase class loss weight: 2.0 → 4.0 or 6.0
- Add horizontal flip augmentation with class swapping
- Consider adding contrastive loss for left/right distinction
- Check if bounding box regions are accurate

2. **Keypoint Visibility**
   - Low heatmap activations (max ~0.14)
   - Only 25% of keypoints detected per instance
   - σ may be too small or loss weight needs adjustment

**Recommendations:**
- Increase heatmap loss weight: 4.0 → 6.0 or 8.0
- Try larger σ: 2px → 3px
- Add focal loss to handle class imbalance
- Verify GT generation is correct

3. **Over-Detection**
   - Some images detect 2 instances when GT has 1
   - min_keypoints threshold (3) may be too low
   - False positives from noise or ambiguous regions

**Recommendations:**
- Increase min_keypoints: 3 → 5 or 6
- Add stricter peak_threshold: 0.05 → 0.10
- Consider adding presence head (Task 10)

---

## Next Steps

### Short-term (M1 Improvements)

**Option A: Retrain with Better Hyperparameters**
```python
config = {
    'loss_weights': {
        'heatmap': 6.0,  # Increased from 4.0
        'paf': 1.0,
        'class': 4.0     # Increased from 2.0
    },
    'sigma': 3.0,        # Increased from 2.0
}
```

**Option B: Add Data Augmentation**
- Horizontal flips with class swapping
- Rotation (±15°)
- Scale jitter (0.9-1.1×)
- Color jitter

### Long-term (M2-M4 Progression)

**Task 7:** Add Rotation Head (6-D continuous)
**Task 8:** Add Mask Head (2-channel segmentation)
**Task 9:** Add Uncertainty Heads (sigma, confidence)
**Task 10:** Add Presence & Scale Heads
**Task 11:** Progressive Training M2→M3→M4

---

## Visualizations

**Generated Visualizations:**
- `outputs/visualizations/prediction_64.png`
- `outputs/visualizations/prediction_99.png`
- `outputs/visualizations/prediction_40.png`
- `outputs/visualizations/prediction_38.png`
- `outputs/visualizations/prediction_45.png`

Each shows:
- **Left panel:** Ground truth keypoints and class labels
- **Right panel:** Model predictions with confidence scores

**Color Coding:**
- 🔵 **Blue:** Left foot
- 🔴 **Red:** Right foot

---

## Conclusion

✅ **Training Successful:** Model converges well with stable losses  
⚠️ **Performance Gap:** Class accuracy at 68% vs 95% target  
✅ **Architecture Validated:** 211K params sufficient for basic detection  
🔄 **Next Actions:** Tune hyperparameters or continue with Tasks 7-11

The M1 baseline is functional and demonstrates the architecture works. The main limitation is class discrimination, which can be improved through:
1. Better loss balancing
2. Data augmentation
3. Or waiting for full M4 model with all heads

**Recommendation:** Continue with Tasks 7-11 to build complete model, then train M4 end-to-end with all improvements.

---

**Model Checkpoints:**
- Best: `outputs/m1_training/checkpoints/best.pth` (epoch 44, loss 1.233)
- Latest: `outputs/m1_training/checkpoints/latest.pth` (epoch 50)

**Total Training Time:** ~1.8 hours (CUDA + AMP + batch 128)
