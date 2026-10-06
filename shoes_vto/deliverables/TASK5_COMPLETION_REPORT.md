# Task 5 Completion Report: Keypoint Grouping Module

**Date:** 2026-09-26  
**Status:** ✅ COMPLETE

---

## Overview

Successfully implemented the keypoint grouping module that converts heatmap peaks, PAF fields, and class predictions into structured foot instances with left/right classification.

---

## Components Implemented

### 1. **Peak Detection (`detect_peaks`)**
- Non-Maximum Suppression (NMS) using scipy `maximum_filter`
- Returns `(x, y, confidence)` tuples sorted by confidence
- Configurable threshold and NMS window size

### 2. **Batch Peak Detection (`detect_all_peaks`)**
- Processes all 16 keypoint channels for batch of images
- Returns dict structure: `{kp_idx: [(x, y, conf), ...]}`
- Supports max peaks per keypoint limit

### 3. **PAF Scoring (`compute_paf_score`)**
- Computes connection score between two keypoint candidates
- Samples PAF vectors along the limb line
- Returns normalized score (0-1) based on alignment

### 4. **Connection Finding (`find_connections`)**
- Tests all peak pairs for a given limb
- Filters by PAF threshold
- Returns sorted connections with combined score: `PAF_score × conf_from × conf_to`

### 5. **Greedy Assembly (`greedy_assembly`)**
- Assembles keypoint instances from peaks and connections
- Uses greedy algorithm: start with strongest connections
- Tracks used peaks to avoid duplicates
- Merges instances when they share keypoints
- Filters by minimum keypoint count
- Returns instances with structure:
  ```python
  {
    'keypoints': [[x, y, conf], ...],  # 16 elements (0 for missing)
    'score': float,  # Overall instance confidence
    'num_keypoints': int  # Count of visible keypoints
  }
  ```

### 6. **Class Assignment (`assign_class_to_instances`)**
- Samples class probability map at keypoint locations
- Votes across all visible keypoints
- Assigns class_id (0=left, 1=right) and confidence
- Adds `class_id` and `class_confidence` to each instance

### 7. **Complete Pipeline (`group_keypoints`)**
- End-to-end batch processing
- Loads PAF configuration automatically
- Returns batch results: `List[List[instance]]`
- Configurable thresholds and limits

---

## File Structure

```
shoes_vto/
├── src/
│   └── utils/
│       └── keypoint_grouping.py         (418 lines) ✅
├── test_task5.py                         (219 lines) ✅
└── configs/
    └── paf_connections.yaml              (loaded automatically)
```

---

## Test Results

**All 10 tests passed ✅**

1. ✅ Peak Detection - Single Peak
2. ✅ Peak Detection - Multiple Peaks  
3. ✅ Peak Detection - NMS
4. ✅ PAF Score - Aligned
5. ✅ PAF Score - Misaligned
6. ✅ Greedy Assembly - Single Instance
7. ✅ Class Assignment
8. ✅ Group Keypoints - Empty
9. ✅ Group Keypoints - Full Pipeline
10. ✅ Integration - All Heads to Grouping

---

## Technical Details

### PAF Connection Format
- 15 limb pairs defined in `paf_connections.yaml`
- Each limb: `[from_idx, to_idx, name]`
- PAF channels: 30 (15 limbs × 2 for x,y components)

### Instance Assembly Algorithm
1. Detect peaks in all 16 heatmap channels
2. For each limb pair, find valid connections using PAF scores
3. Sort all connections by combined score
4. Greedily merge connections into instances:
   - Start with strongest connection
   - Merge instances that share keypoints
   - Track used peaks to avoid duplicates
5. Filter instances by minimum keypoint count
6. Assign left/right class by voting across keypoints
7. Return top N instances sorted by score

### Key Design Decisions

**Peak Format:** `(x, y, confidence)` tuples
- Consistent with standard pose estimation libraries
- Confidence enables weighted voting and scoring

**Greedy Merging:**
- Simpler than Hungarian algorithm
- Sufficient for max 2 feet per frame
- Faster inference (critical for 7ms budget)

**Class Voting:**
- Averages class probabilities across all visible keypoints
- Robust to individual keypoint misclassifications
- Confidence reflects vote agreement

**Config-Driven:**
- PAF limb pairs loaded from YAML
- Easy to modify connection structure
- No hardcoded indices in core algorithm

---

## Integration Points

### Inputs (from model heads):
- `heatmaps: [B, 16, 64, 64]` - from HeatmapHead
- `pafs: [B, 30, 64, 64]` - from PAFHead  
- `class_probs: [B, 2, 64, 64]` - from ClassHead (softmax)

### Outputs (structured instances):
```python
[
  [  # Batch item 0
    {
      'keypoints': [[x, y, conf], ...],  # 16 keypoints
      'score': 2.53,
      'num_keypoints': 14,
      'class_id': 0,  # 0=left, 1=right
      'class_confidence': 0.94
    },
    {...}  # Instance 2 (if present)
  ],
  [...]  # Batch item 1
]
```

---

## Performance Characteristics

- **Peak detection:** O(H × W) per channel with NMS
- **PAF scoring:** O(num_samples) per peak pair
- **Assembly:** O(num_connections × num_instances) greedy merge
- **Memory:** Minimal allocations, mostly in-place operations
- **Parallelization:** Batch dimension independent

**Expected inference contribution:** <0.5ms (CPU post-processing)

---

## Known Limitations

1. **Max instances:** Currently limited to 2 per image (configurable)
2. **Greedy assembly:** May not find globally optimal solution
3. **No temporal tracking:** Each frame processed independently
4. **Fixed thresholds:** Not adaptive to scene conditions

These limitations are acceptable for M1 baseline. Can be improved in M2/M3.

---

## Next Steps (Task 6)

Now that keypoint grouping is complete, Task 6 will:
1. Create complete model class combining encoder + all heads
2. Implement training loop with all loss functions
3. Add data loading and augmentation pipeline
4. Train M1 baseline model on 880 training images
5. Validate on 110 validation images
6. Report metrics: mAP@50, class accuracy, convergence

---

## Files Modified/Created

### Created:
- `shoes_vto/src/utils/keypoint_grouping.py` (418 lines)
- `shoes_vto/test_task5.py` (219 lines)
- `shoes_vto/deliverables/TASK5_COMPLETION_REPORT.md` (this file)

### Modified:
- `README.md` - Updated Task 5 status to DONE
- `shoes_vto/src/utils/keypoint_grouping.py` - Fixed config path resolution

---

## Verification Checklist

- [x] All functions implement expected signatures
- [x] Peak detection with NMS working correctly
- [x] PAF scoring validates alignment
- [x] Greedy assembly merges instances properly  
- [x] Class assignment votes across keypoints
- [x] Full pipeline integrates all components
- [x] Config loading robust to different working directories
- [x] All 10 tests pass
- [x] Integration test confirms heads→grouping works end-to-end
- [x] No gradients required (inference-only module)
- [x] README updated
- [x] Completion report created

---

**Task 5 is complete and ready for integration into training pipeline (Task 6).**
