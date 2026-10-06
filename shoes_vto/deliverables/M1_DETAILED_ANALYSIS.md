# ARShoe M1 - Detailed Problem Analysis

**Date:** 2026-09-26  
**Analysis:** Raw model outputs vs ground truth on 5 test images

---

## 🔴 CRITICAL ISSUES IDENTIFIED

###  1. **Keypoint Localization is VERY POOR**

**Problem:** Predicted heatmap peaks are 7-56 pixels away from ground truth locations!

**Evidence from Test Images:**

| Image | Keypoint | GT Location | Pred Peak Distance | Status |
|-------|----------|-------------|-------------------|---------|
| 64 | KP 0 | High confidence | 19.1 px away | ❌ BAD |
| 64 | KP 2 | High confidence | 43.0 px away | ❌ BAD |
| 99 | KP 0 | High confidence | 27.1 px away | ❌ BAD |
| 40 | KP 4 | High confidence | 23.6 px away | ❌ BAD |
| 38 | KP 3 | High confidence | **2.0 px away** | ✅ GOOD (rare!) |
| 45 | KP 0 | High confidence | 56.3 px away | ❌ TERRIBLE |

**Key Observations:**
- **Average distance error: 20-30 pixels** (on 64×64 heatmap = 80-120px on 256×256 image!)
- Only 4 out of 72 keypoints have distance < 3px
- Most keypoints are 15-50px away from GT location
- **The model is learning keypoint existence but not precise location**

**Root Cause:**
```
GT Value at location: 0.9-1.0 (perfect Gaussian peak)
Pred Value at GT location: 0.03-0.08 (very low!)
Pred Max elsewhere: 0.08-0.14 (also low but higher)
```

The model is predicting weak, diffuse heatmaps instead of sharp peaks.

---

### 2. **Heatmap Activations Are TOO LOW**

**Problem:** Predicted heatmap values are 10-20x lower than they should be.

**Evidence:**

| Metric | Ground Truth | Predicted | Ratio |
|--------|-------------|-----------|-------|
| **Max Value** | 0.95-1.00 | 0.08-0.14 | **7-12x lower** |
| **Value at GT location** | 0.85-1.00 | 0.03-0.08 | **12-30x lower** |
| **Mean (visible regions)** | ~0.5 | 0.037-0.048 | **10-13x lower** |

**Why This Happens:**
1. **Sigmoid activation** caps values at [0, 1]
2. **MSE loss** doesn't penalize low confidence enough
3. Model learns to play it safe with low activations
4. **Gaussian σ=2px** might be too tight on 64×64 map

---

### 3. **Class Prediction is WRONG 40-70% of the Time**

**Problem:** Model frequently confuses left and right feet.

**Per-Image Analysis:**

| Image | GT Classes | Predictions | Pixel Accuracy | Issues |
|-------|------------|-------------|----------------|---------|
| 64 | Right + Left | **Left** + Left | 66.6% | Right misclassified as Left |
| 99 | Right | **Left** | 29.2% | Completely wrong |
| 40 | Right | **Left** | 53.3% | Completely wrong |
| 38 | Left + Right | Left + Right | 65.8% | ✅ Correct! |
| 45 | Right + Left | Right + **Right** | 65.2% | Left misclassified as Right |

**Pattern:**
- **3/5 images** (60%) have misclassified instances
- Right feet are misclassified more often than left
- Right foot accuracy: 29-58% (terrible!)
- Left foot accuracy: 72-79% (better but not good enough)

**Root Cause:**
- Class loss weight (2.0) is too low compared to heatmap (4.0)
- Model prioritizes keypoint detection over class discrimination
- No data augmentation (horizontal flips) to teach left/right distinction
- Class head may be undertrained

---

## 📊 Quantitative Summary

### Heatmap Performance
```
MSE Loss: 0.004 (seems good but misleading!)
  
Actual Performance:
- Distance Error: 20-30px average (TERRIBLE for 64×64 map)
- Peak Sharpness: 7-12x weaker than GT
- Localization Accuracy: <6% within 3px tolerance
```

### Class Performance
```
Training Metric: 67.9% accuracy

Actual Performance:
- Per-pixel accuracy: 29-72% (highly variable)
- Instance-level: 40% wrong (3/5 test images)
- Right foot bias: Frequently predicted as left
```

---

## 🔍 Root Cause Analysis

### Why is Heatmap Localization So Bad?

**1. Gaussian σ=2px is TOO SMALL**
- On 64×64 map, 2px Gaussian covers only ~13×13 pixels
- Very narrow, easy to miss
- Model learns general region but not precise peak

**2. MSE Loss Doesn't Penalize Diffuse Predictions**
```python
GT:   [0, 0, 0, 1.0, 0, 0]  # Sharp peak
Pred: [0.05, 0.08, 0.12, 0.15, 0.11, 0.07]  # Diffuse blob

MSE = 0.03  # Looks small!
But peak is 13 pixels away and only 0.15 instead of 1.0
```

**3. Heatmap Loss Weight May Not Be Enough**
- Current: 4.0 × HM + 1.0 × PAF + 2.0 × Class
- MSE with diffuse predictions still gives low loss
- Model doesn't feel pressure to sharpen peaks

**4. No Peak-Aware Loss Function**
- MSE treats all pixels equally
- Doesn't specifically reward correct peak location
- Focal loss or adaptive wing loss might help

### Why is Class Prediction So Bad?

**1. Class Loss Weight Too Low**
- Currently 2.0 vs 4.0 for heatmap
- Model learns "keypoints exist" before "which foot"
- By epoch 50, keypoint patterns are locked in

**2. No Left/Right Data Augmentation**
- No horizontal flips with class swapping
- Model never learns to distinguish mirrored poses
- Training data may have imbalanced left/right distribution

**3. Class Map Generation May Be Suboptimal**
- Uses bbox expansion (10px)
- May include ambiguous regions
- Might benefit from tighter, keypoint-based regions

---

## ✅ SOLUTIONS - Ranked by Impact

### 🔥 HIGH IMPACT (Do These First)

#### Solution 1: Increase Gaussian σ
```python
# Current
sigma = 2.0  # Too tight

# Recommended
sigma = 3.5 or 4.0  # Wider, easier to learn
```
**Expected improvement:** +50-70% localization accuracy

#### Solution 2: Use Focal Loss for Heatmaps
```python
# Replace MSE with Focal Loss
class FocalHeatmapLoss:
    alpha = 2.0  # Focus on hard examples
    beta = 4.0   # Penalize false negatives more
```
**Expected improvement:** +30-40% peak sharpness

#### Solution 3: Dramatically Increase Class Loss Weight
```python
loss_weights = {
    'heatmap': 4.0,
    'paf': 1.0,
    'class': 6.0  # Increased from 2.0
}
```
**Expected improvement:** +20-30% class accuracy

#### Solution 4: Add Horizontal Flip Augmentation
```python
# In data loader
if random.random() > 0.5:
    image = torch.flip(image, dims=[2])  # Horizontal flip
    keypoints[:, 0] = 256 - keypoints[:, 0]  # Mirror x coords
    class_id = 1 - class_id  # Swap left<->right
```
**Expected improvement:** +15-25% class accuracy

### ⚡ MEDIUM IMPACT

#### Solution 5: Adjust Loss Balance
```python
loss_weights = {
    'heatmap': 6.0,  # Increased from 4.0
    'paf': 1.0,
    'class': 6.0     # Increased from 2.0
}
```

#### Solution 6: Add Peak Location Loss
```python
# Additional loss term
def peak_location_loss(pred_heatmaps, gt_heatmaps):
    pred_peaks = get_peak_locations(pred_heatmaps)
    gt_peaks = get_peak_locations(gt_heatmaps)
    return mse(pred_peaks, gt_peaks)
```

#### Solution 7: Increase Training Epochs
- Current: 50 epochs
- Try: 100 epochs with better patience

### 🔧 LOW IMPACT (Nice to Have)

- Learning rate warmup
- Cosine annealing schedule
- Label smoothing for class
- Different backbone architecture

---

## 🎯 Recommended Training Plan

### Quick Fix (2-3 hours training)
```python
config = {
    'sigma': 3.5,  # Wider Gaussians
    'loss_weights': {
        'heatmap': 6.0,  # Increased
        'paf': 1.0,
        'class': 6.0     # Increased
    },
    'num_epochs': 50,
    'use_focal_loss': True,
    'horizontal_flip': True
}
```
**Expected result:** 
- Keypoint localization: 70-80% within 5px
- Class accuracy: 85-90%

### Better Fix (4-5 hours training)
Add all high-impact solutions + train for 100 epochs

**Expected result:**
- Keypoint localization: 85-90% within 3px
- Class accuracy: 93-95%
- mAP@50: 0.75-0.80

---

## 📈 Comparison: Current vs Expected

| Metric | Current M1 | After Quick Fix | After Better Fix | Target |
|--------|------------|-----------------|------------------|--------|
| **Heatmap Peak Distance** | 20-30px | 3-5px | 1-3px | <3px |
| **Heatmap Max Value** | 0.08-0.14 | 0.4-0.6 | 0.7-0.9 | >0.7 |
| **Class Accuracy (pixel)** | 29-72% | 85-90% | 93-95% | >95% |
| **Class Accuracy (instance)** | 40% wrong | 10-15% wrong | <5% wrong | >95% |
| **Training Loss** | 1.13 | 0.8-0.9 | 0.6-0.7 | <0.7 |
| **mAP@50** | ~0.30* | ~0.75 | ~0.85 | >0.80 |

*Estimated based on poor localization

---

## 🚀 Next Steps

1. **Implement Quick Fix** (recommended start):
   - Update `generate_heatmaps_batch`: σ = 2.0 → 3.5
   - Update `train_m1.py`: class weight = 2.0 → 6.0
   - Add horizontal flip in dataset loader
   - Retrain for 50 epochs

2. **Evaluate Results**:
   - Run `evaluate_detailed.py` again
   - Check if distance errors drop to <5px
   - Verify class accuracy improves to >85%

3. **If Still Not Good Enough**:
   - Implement focal loss for heatmaps
   - Train for 100 epochs
   - Consider adding peak location loss

4. **Once M1 Works Well**:
   - Continue with Tasks 7-11 (remaining heads)
   - Train complete M4 model

---

## 💡 Key Insight

**The low training loss (1.13) was misleading!**

MSE loss can be low even when:
- Peaks are in wrong locations (diffuse blobs)
- Values are too low (0.15 instead of 1.0)
- Classes are wrong (50% accuracy still gives reasonable loss)

**Lesson:** Always evaluate on actual metrics (peak distance, class accuracy) not just loss values.

---

**Recommendation:** Start with Quick Fix. It addresses the two biggest issues (localization and classification) with minimal code changes. Expected training time: ~2 hours on your GPU.
