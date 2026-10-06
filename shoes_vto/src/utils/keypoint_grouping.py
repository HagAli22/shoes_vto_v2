"""
Keypoint Grouping with PAF and Class Integration

Groups detected keypoint peaks into coherent instances using:
1. Heatmap peak detection (NMS)
2. PAF-based limb connection scoring
3. Greedy instance assembly
4. Class head disambiguation for left/right

This is the core post-processing that converts network outputs to detected foot instances.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from scipy.ndimage import maximum_filter


def load_paf_config():
    """Load PAF limb connection configuration"""
    # Try multiple paths to find the config
    possible_paths = [
        Path(__file__).parent.parent.parent / "configs" / "paf_connections.yaml",  # From src/utils/
        Path("shoes_vto/configs/paf_connections.yaml"),  # From project root
        Path("configs/paf_connections.yaml"),  # From shoes_vto/
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
    
    raise FileNotFoundError(f"Could not find paf_connections.yaml in any of: {possible_paths}")


def detect_peaks(heatmap, threshold=0.1, nms_window=3):
    """
    Detect keypoint peaks in heatmap using Non-Maximum Suppression
    
    Args:
        heatmap: [H, W] single keypoint heatmap
        threshold: Minimum confidence threshold
        nms_window: Window size for NMS (e.g., 3 means 3×3)
    
    Returns:
        peaks: List of (x, y, confidence) tuples
    """
    # Convert to numpy if tensor
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    # Apply NMS using maximum filter
    local_max = maximum_filter(heatmap, size=nms_window)
    peaks_binary = (heatmap == local_max) & (heatmap > threshold)
    
    # Get peak coordinates
    peak_coords = np.where(peaks_binary)
    
    peaks = []
    for i in range(len(peak_coords[0])):
        y = peak_coords[0][i]
        x = peak_coords[1][i]
        confidence = heatmap[y, x]
        peaks.append((x, y, confidence))
    
    # Sort by confidence (descending)
    peaks = sorted(peaks, key=lambda p: p[2], reverse=True)
    
    return peaks


def detect_all_peaks(heatmaps, threshold=0.1, nms_window=3, max_peaks_per_kp=10):
    """
    Detect peaks for all keypoints in batch
    
    Args:
        heatmaps: [B, 16, H, W] heatmaps
        threshold: Minimum confidence threshold
        nms_window: NMS window size
        max_peaks_per_kp: Maximum peaks to keep per keypoint
    
    Returns:
        all_peaks: List of length B, each containing dict:
            {kp_idx: [(x, y, confidence), ...]}
    """
    batch_size, num_kps, height, width = heatmaps.shape
    
    batch_peaks = []
    
    for b in range(batch_size):
        image_peaks = {}
        
        for kp_idx in range(num_kps):
            heatmap = heatmaps[b, kp_idx]
            peaks = detect_peaks(heatmap, threshold, nms_window)
            
            # Keep top N peaks
            peaks = peaks[:max_peaks_per_kp]
            
            image_peaks[kp_idx] = peaks
        
        batch_peaks.append(image_peaks)
    
    return batch_peaks


def compute_paf_score(paf_x, paf_y, point1, point2, num_samples=10):
    """
    Compute PAF connection score between two keypoints
    
    Args:
        paf_x: [H, W] PAF x-component
        paf_y: [H, W] PAF y-component
        point1: (x1, y1) starting point
        point2: (x2, y2) ending point
        num_samples: Number of points to sample along the line
    
    Returns:
        score: PAF connection score (0-1, higher is better)
    """
    x1, y1 = point1
    x2, y2 = point2
    
    # Vector from point1 to point2
    vec = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    vec_length = np.linalg.norm(vec)
    
    if vec_length < 1e-6:
        return 0.0
    
    # Unit vector
    vec_unit = vec / vec_length
    
    # Sample points along the line
    scores = []
    for i in range(num_samples):
        t = i / (num_samples - 1) if num_samples > 1 else 0.5
        
        # Interpolated point
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        
        # Clip to valid range
        height, width = paf_x.shape
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        
        # Get PAF vector at this point
        paf_vec = np.array([paf_x[y, x], paf_y[y, x]], dtype=np.float32)
        
        # Dot product with unit vector
        score = np.dot(paf_vec, vec_unit)
        scores.append(score)
    
    # Average score along the line
    avg_score = np.mean(scores)
    
    # Normalize to [0, 1] (assuming PAF vectors are unit length)
    score = max(0.0, avg_score)
    
    return score


def find_connections(peaks_from, peaks_to, pafs, limb_idx, paf_threshold=0.05):
    """
    Find connections between two sets of keypoint peaks using PAF
    
    Args:
        peaks_from: List of (x, y, conf) for starting keypoint
        peaks_to: List of (x, y, conf) for ending keypoint
        pafs: [30, H, W] PAF fields
        limb_idx: Index of limb (0-14)
        paf_threshold: Minimum PAF score threshold
    
    Returns:
        connections: List of (from_idx, to_idx, score) tuples
    """
    # Get PAF channels for this limb
    paf_x = pafs[limb_idx * 2].cpu().numpy()
    paf_y = pafs[limb_idx * 2 + 1].cpu().numpy()
    
    connections = []
    
    # Try all combinations
    for i, peak_from in enumerate(peaks_from):
        for j, peak_to in enumerate(peaks_to):
            # Compute PAF score
            score = compute_paf_score(
                paf_x, paf_y,
                (peak_from[0], peak_from[1]),
                (peak_to[0], peak_to[1])
            )
            
            if score > paf_threshold:
                # Combined score: PAF score × peak confidences
                combined_score = score * peak_from[2] * peak_to[2]
                connections.append((i, j, combined_score))
    
    # Sort by score (descending)
    connections = sorted(connections, key=lambda c: c[2], reverse=True)
    
    return connections


def greedy_assembly(all_peaks, pafs, limb_pairs, paf_threshold=0.05, min_keypoints=8):
    """
    Greedily assemble instances from peaks and PAF connections
    
    Args:
        all_peaks: Dict {kp_idx: [(x, y, conf), ...]}
        pafs: [30, H, W] PAF fields
        limb_pairs: List of [from_idx, to_idx, name]
        paf_threshold: Minimum PAF score
        min_keypoints: Minimum keypoints required for an instance
    
    Returns:
        instances: List of instances, each is:
            {'keypoints': [[x, y, conf], ...] for 16 KPs,
             'score': overall confidence}
    """
    num_keypoints = 16
    
    # Initialize instances storage
    instances = []
    used_peaks = {kp_idx: set() for kp_idx in range(num_keypoints)}
    
    # Build connection graph
    connection_graph = {}
    
    for limb_idx, (from_idx, to_idx, name) in enumerate(limb_pairs):
        if from_idx not in all_peaks or to_idx not in all_peaks:
            continue
        
        peaks_from = all_peaks[from_idx]
        peaks_to = all_peaks[to_idx]
        
        if len(peaks_from) == 0 or len(peaks_to) == 0:
            continue
        
        # Find connections
        connections = find_connections(peaks_from, peaks_to, pafs, limb_idx, paf_threshold)
        
        connection_graph[(from_idx, to_idx)] = connections
    
    # Greedy assembly: start from strongest connections
    # Collect all connections across all limbs
    all_connections = []
    for (from_idx, to_idx), connections in connection_graph.items():
        for from_peak_idx, to_peak_idx, score in connections:
            all_connections.append((from_idx, to_idx, from_peak_idx, to_peak_idx, score))
    
    # Sort by score
    all_connections = sorted(all_connections, key=lambda c: c[4], reverse=True)
    
    # Build instances greedily
    instance_keypoints = []
    
    for from_kp_idx, to_kp_idx, from_peak_idx, to_peak_idx, score in all_connections:
        # Check if these peaks are already used
        if from_peak_idx in used_peaks[from_kp_idx] and to_peak_idx in used_peaks[to_kp_idx]:
            # Both already in same instance, skip
            continue
        
        # Find or create instance
        from_peak = all_peaks[from_kp_idx][from_peak_idx]
        to_peak = all_peaks[to_kp_idx][to_peak_idx]
        
        # Try to find existing instance with either peak
        found_instance = None
        for inst in instance_keypoints:
            if (inst.get(from_kp_idx) is not None and 
                inst[from_kp_idx] == from_peak_idx):
                found_instance = inst
                break
            if (inst.get(to_kp_idx) is not None and 
                inst[to_kp_idx] == to_peak_idx):
                found_instance = inst
                break
        
        if found_instance is None:
            # Create new instance
            new_instance = {from_kp_idx: from_peak_idx, to_kp_idx: to_peak_idx}
            instance_keypoints.append(new_instance)
            used_peaks[from_kp_idx].add(from_peak_idx)
            used_peaks[to_kp_idx].add(to_peak_idx)
        else:
            # Add to existing instance
            if from_kp_idx not in found_instance:
                found_instance[from_kp_idx] = from_peak_idx
                used_peaks[from_kp_idx].add(from_peak_idx)
            if to_kp_idx not in found_instance:
                found_instance[to_kp_idx] = to_peak_idx
                used_peaks[to_kp_idx].add(to_peak_idx)
    
    # Convert to final format
    final_instances = []
    
    for inst in instance_keypoints:
        # Check minimum keypoints
        if len(inst) < min_keypoints:
            continue
        
        # Build keypoint array
        keypoints = []
        scores = []
        
        for kp_idx in range(num_keypoints):
            if kp_idx in inst:
                peak_idx = inst[kp_idx]
                x, y, conf = all_peaks[kp_idx][peak_idx]
                keypoints.append([x, y, conf])
                scores.append(conf)
            else:
                keypoints.append([0, 0, 0])  # Missing keypoint
        
        # Overall instance score
        instance_score = np.mean(scores) if len(scores) > 0 else 0.0
        
        final_instances.append({
            'keypoints': keypoints,
            'score': instance_score,
            'num_keypoints': len(inst)
        })
    
    return final_instances


def assign_class_to_instances(instances, class_probs, heatmap_size=64):
    """
    Assign left/right class to each instance using class head output
    
    Args:
        instances: List of instances with keypoints
        class_probs: [2, H, W] class probability map
        heatmap_size: Size of heatmap/class map
    
    Returns:
        instances: Same list with 'class_id' and 'class_confidence' added
    """
    for instance in instances:
        keypoints = instance['keypoints']
        
        # Sample class probabilities at keypoint locations
        votes = []
        
        for kp in keypoints:
            x, y, conf = kp
            if conf > 0:  # Only visible keypoints
                # Convert to heatmap coordinates
                hm_x = int(x)
                hm_y = int(y)
                
                # Clip to valid range
                hm_x = max(0, min(heatmap_size - 1, hm_x))
                hm_y = max(0, min(heatmap_size - 1, hm_y))
                
                # Get class probabilities
                left_prob = class_probs[0, hm_y, hm_x].item()
                right_prob = class_probs[1, hm_y, hm_x].item()
                
                votes.append((left_prob, right_prob))
        
        if len(votes) == 0:
            # No visible keypoints, default to left
            instance['class_id'] = 0
            instance['class_confidence'] = 0.5
        else:
            # Average votes
            avg_left = sum(v[0] for v in votes) / len(votes)
            avg_right = sum(v[1] for v in votes) / len(votes)
            
            # Determine class
            if avg_left > avg_right:
                instance['class_id'] = 0
                instance['class_confidence'] = avg_left
            else:
                instance['class_id'] = 1
                instance['class_confidence'] = avg_right
    
    return instances


def group_keypoints(heatmaps, pafs, class_probs, 
                    peak_threshold=0.1, 
                    paf_threshold=0.05, 
                    nms_window=3,
                    min_keypoints=8,
                    max_instances=2):
    """
    Complete keypoint grouping pipeline
    
    Args:
        heatmaps: [B, 16, H, W] keypoint heatmaps
        pafs: [B, 30, H, W] PAF fields
        class_probs: [B, 2, H, W] class probability maps
        peak_threshold: Minimum peak confidence
        paf_threshold: Minimum PAF score
        nms_window: NMS window size
        min_keypoints: Minimum keypoints per instance
        max_instances: Maximum instances to return per image
    
    Returns:
        batch_instances: List of length B, each containing list of instances
    """
    # Load PAF configuration
    config = load_paf_config()
    limb_pairs = config['limb_pairs']
    
    batch_size = heatmaps.shape[0]
    batch_instances = []
    
    for b in range(batch_size):
        # Detect peaks
        all_peaks = {}
        for kp_idx in range(16):
            heatmap = heatmaps[b, kp_idx]
            peaks = detect_peaks(heatmap, peak_threshold, nms_window)
            all_peaks[kp_idx] = peaks
        
        # Assemble instances
        instances = greedy_assembly(
            all_peaks, 
            pafs[b], 
            limb_pairs, 
            paf_threshold, 
            min_keypoints
        )
        
        # Assign classes
        instances = assign_class_to_instances(instances, class_probs[b])
        
        # Sort by score and keep top N
        instances = sorted(instances, key=lambda i: i['score'], reverse=True)
        instances = instances[:max_instances]
        
        batch_instances.append(instances)
    
    return batch_instances


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Keypoint Grouping")
    print("=" * 80)
    
    # Test 1: Peak detection
    print("\n[Test 1] Peak detection with NMS")
    
    # Create synthetic heatmap with 3 peaks
    heatmap = torch.zeros(64, 64)
    heatmap[20, 20] = 0.9
    heatmap[40, 40] = 0.7
    heatmap[50, 10] = 0.5
    
    # Add some noise
    heatmap += torch.rand(64, 64) * 0.1
    
    peaks = detect_peaks(heatmap, threshold=0.3, nms_window=3)
    
    print(f"  Number of peaks detected: {len(peaks)}")
    print(f"  Expected: 3")
    print(f"  Peaks: {peaks[:5]}")
    
    assert len(peaks) == 3, f"Expected 3 peaks, got {len(peaks)}"
    print("  ✅ PASS")
    
    # Test 2: Batch peak detection
    print("\n[Test 2] Batch peak detection")
    
    batch_heatmaps = torch.rand(2, 16, 64, 64) * 0.3
    # Add some strong peaks
    batch_heatmaps[0, 0, 30, 30] = 0.8
    batch_heatmaps[0, 1, 35, 35] = 0.7
    batch_heatmaps[1, 0, 20, 20] = 0.9
    
    batch_peaks = detect_all_peaks(batch_heatmaps, threshold=0.5, max_peaks_per_kp=5)
    
    print(f"  Batch size: {len(batch_peaks)}")
    print(f"  Image 0, KP 0 peaks: {len(batch_peaks[0][0])}")
    print(f"  Image 1, KP 0 peaks: {len(batch_peaks[1][0])}")
    
    assert len(batch_peaks) == 2, "Batch size mismatch"
    print("  ✅ PASS")
    
    # Test 3: PAF score computation
    print("\n[Test 3] PAF score computation")
    
    # Create synthetic PAF field
    paf_x = torch.zeros(64, 64)
    paf_y = torch.zeros(64, 64)
    
    # Create a vector field from (10, 10) to (50, 50)
    for i in range(15, 45):
        for j in range(15, 45):
            paf_x[j, i] = 0.7  # x-component (positive = right)
            paf_y[j, i] = 0.7  # y-component (positive = down)
    
    # Test connection along the field
    score = compute_paf_score(
        paf_x.numpy(), paf_y.numpy(),
        (10, 10), (50, 50),
        num_samples=10
    )
    
    print(f"  PAF score (aligned): {score:.3f}")
    print(f"  Expected: > 0.5 (aligned with field)")
    
    # Test connection perpendicular to field
    score_perp = compute_paf_score(
        paf_x.numpy(), paf_y.numpy(),
        (10, 50), (50, 10),
        num_samples=10
    )
    
    print(f"  PAF score (perpendicular): {score_perp:.3f}")
    print(f"  Expected: < 0.3 (perpendicular to field)")
    
    assert score > 0.5, "Aligned connection should have high score"
    print("  ✅ PASS")
    
    # Test 4: Load PAF config
    print("\n[Test 4] Load PAF configuration")
    
    try:
        config = load_paf_config()
        limb_pairs = config['limb_pairs']
        
        print(f"  Number of limbs: {len(limb_pairs)}")
        print(f"  Expected: 15")
        print(f"  First limb: {limb_pairs[0]}")
        
        assert len(limb_pairs) == 15, "Should have 15 limb pairs"
        print("  ✅ PASS")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
    
    print("\n" + "=" * 80)
    print("Keypoint grouping tests complete!")
    print("Note: Full integration test in test_task5.py")
    print("=" * 80)
