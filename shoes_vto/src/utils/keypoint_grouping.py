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
import cv2
import yaml
import json
from collections import defaultdict
from pathlib import Path
from scipy.ndimage import maximum_filter


def load_paf_config(config_name="paf_connections.yaml"):
    """Load PAF limb connection configuration"""
    if isinstance(config_name, (str, Path)) and Path(config_name).is_file():
        with open(config_name, 'r') as f:
            return yaml.safe_load(f)
            
    # Try multiple paths to find the config
    fname = Path(config_name).name
    possible_paths = [
        Path(__file__).parent.parent.parent / "configs" / fname,  # From src/utils/
        Path("shoes_vto/configs") / fname,  # From project root
        Path("configs") / fname,  # From shoes_vto/
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
    
    raise FileNotFoundError(f"Could not find paf_connections.yaml in any of: {possible_paths}")


def load_skeleton_priors(config_name="skeleton_priors_train.json"):
    """Load learned pairwise skeleton distance priors from training set"""
    fname = Path(config_name).name
    possible_paths = [
        Path(__file__).parent.parent.parent / "configs" / fname,
        Path("shoes_vto/configs") / fname,
        Path("configs") / fname,
    ]
    for config_path in possible_paths:
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
    return {}


def detect_peaks(heatmap, threshold=0.1, nms_window=3, subpixel=True):
    """
    Detect keypoint peaks in heatmap using Non-Maximum Suppression with optional sub-pixel refinement.
    
    Args:
        heatmap: [H, W] single keypoint heatmap
        threshold: Minimum confidence threshold
        nms_window: Window size for NMS (e.g., 3 means 3×3)
        subpixel: Whether to apply quadratic Taylor expansion sub-pixel refinement
    
    Returns:
        peaks: List of (x, y, confidence) tuples
    """
    # Convert to numpy if tensor
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    
    # Apply NMS using maximum filter
    local_max = maximum_filter(heatmap, size=nms_window)
    peaks_binary = (heatmap == local_max) & (heatmap > threshold)
    
    # Get peak coordinates
    peak_coords = np.where(peaks_binary)
    h, w = heatmap.shape
    
    peaks = []
    for i in range(len(peak_coords[0])):
        y = int(peak_coords[0][i])
        x = int(peak_coords[1][i])
        confidence = float(heatmap[y, x])
        
        # Sub-pixel quadratic Taylor expansion refinement
        sub_x, sub_y = float(x), float(y)
        if subpixel and 1 <= x < w - 1 and 1 <= y < h - 1:
            # First and second derivatives along x
            dx = 0.5 * (heatmap[y, x + 1] - heatmap[y, x - 1])
            dxx = heatmap[y, x + 1] - 2.0 * heatmap[y, x] + heatmap[y, x - 1]
            if abs(dxx) > 1e-5:
                offset_x = -dx / dxx
                sub_x += float(np.clip(offset_x, -0.5, 0.5))
                
            # First and second derivatives along y
            dy = 0.5 * (heatmap[y + 1, x] - heatmap[y - 1, x])
            dyy = heatmap[y + 1, x] - 2.0 * heatmap[y, x] + heatmap[y - 1, x]
            if abs(dyy) > 1e-5:
                offset_y = -dy / dyy
                sub_y += float(np.clip(offset_y, -0.5, 0.5))
        
        peaks.append((sub_x, sub_y, confidence))
    
    # Sort by confidence (descending)
    peaks = sorted(peaks, key=lambda p: p[2], reverse=True)
    
    return peaks


def detect_all_peaks(heatmaps, threshold=0.1, nms_window=3, max_peaks_per_kp=10, subpixel=True):
    """
    Detect peaks for all keypoints in batch
    
    Args:
        heatmaps: [B, 16, H, W] heatmaps
        threshold: Minimum confidence threshold
        nms_window: NMS window size
        max_peaks_per_kp: Maximum peaks to keep per keypoint
        subpixel: Whether to apply sub-pixel refinement
    
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
            peaks = detect_peaks(heatmap, threshold, nms_window, subpixel=subpixel)
            
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
def compute_paf_score(paf_x, paf_y, point1, point2, num_samples=6):
    """
    Compute PAF score along line segment between two keypoints using vectorized numpy operations.
    
    Args:
        paf_x: [H, W] PAF x-component numpy array
        paf_y: [H, W] PAF y-component numpy array
        point1: (x1, y1) starting point
        point2: (x2, y2) ending point
        num_samples: Number of points to sample along the line
    
    Returns:
        score: PAF connection score (0-1, higher is better)
    """
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    vec_len = (dx * dx + dy * dy) ** 0.5
    
    if vec_len < 1e-6:
        return 0.0
    
    ux = dx / vec_len
    uy = dy / vec_len
    
    t = np.linspace(0, 1, num_samples, dtype=np.float32)
    xs = np.clip(np.round(x1 + t * dx).astype(int), 0, paf_x.shape[1] - 1)
    ys = np.clip(np.round(y1 + t * dy).astype(int), 0, paf_x.shape[0] - 1)
    
    dot_prods = paf_x[ys, xs] * ux + paf_y[ys, xs] * uy
    return float(max(0.0, np.mean(dot_prods)))


def find_connections(peaks_from, peaks_to, paf_x, paf_y, paf_threshold=0.05, max_limb_dist=24.0):
    """
    Find connections between two sets of keypoint peaks using PAF
    
    Args:
        peaks_from: List of (x, y, conf) for starting keypoint
        peaks_to: List of (x, y, conf) for ending keypoint
        paf_x: [H, W] PAF x channel
        paf_y: [H, W] PAF y channel
        paf_threshold: Minimum PAF score threshold
        max_limb_dist: Maximum physical distance between two keypoints on 64x64 grid
    
    Returns:
        connections: List of (from_idx, to_idx, score) tuples
    """
    connections = []
    
    for i, peak_from in enumerate(peaks_from):
        for j, peak_to in enumerate(peaks_to):
            dx = peak_from[0] - peak_to[0]
            dy = peak_from[1] - peak_to[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > max_limb_dist:
                continue
                
            score = compute_paf_score(
                paf_x, paf_y,
                (peak_from[0], peak_from[1]),
                (peak_to[0], peak_to[1])
            )
            
            if score > paf_threshold:
                combined_score = score * peak_from[2] * peak_to[2]
                connections.append((i, j, combined_score))
    
    connections = sorted(connections, key=lambda c: c[2], reverse=True)
    return connections


def greedy_assembly(all_peaks, pafs, limb_pairs, paf_threshold=0.05, min_keypoints=8):
    """
    Greedily assemble instances from peaks and PAF connections
    
    Args:
        all_peaks: Dict {kp_idx: [(x, y, conf), ...]}
        pafs: [30, H, W] PAF fields (Tensor or numpy array)
        limb_pairs: List of [from_idx, to_idx, name]
        paf_threshold: Minimum PAF score
        min_keypoints: Minimum keypoints required for an instance
    
    Returns:
        instances: List of instances
    """
    num_keypoints = 16
    instances = []
    used_peaks = {kp_idx: set() for kp_idx in range(num_keypoints)}
    
    # Convert PAFs to numpy once
    pafs_np = pafs.detach().cpu().numpy() if isinstance(pafs, torch.Tensor) else pafs
    
    connection_graph = {}
    for limb_idx, (from_idx, to_idx, name) in enumerate(limb_pairs):
        if from_idx not in all_peaks or to_idx not in all_peaks:
            continue
        
        peaks_from = all_peaks[from_idx]
        peaks_to = all_peaks[to_idx]
        
        if len(peaks_from) == 0 or len(peaks_to) == 0:
            continue
        
        paf_x = pafs_np[limb_idx * 2]
        paf_y = pafs_np[limb_idx * 2 + 1]
        
        connections = find_connections(peaks_from, peaks_to, paf_x, paf_y, paf_threshold)
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


def load_box_calibration():
    """Load calibrated box margins computed from the training split."""
    calib_paths = [
        Path(__file__).parent.parent.parent / "configs" / "box_calibration.json",
        Path("shoes_vto/configs/box_calibration.json"),
        Path("configs/box_calibration.json"),
    ]
    for cp in calib_paths:
        if cp.exists():
            try:
                import json
                with open(cp, "r") as f:
                    return json.load(f)
            except Exception:
                pass
    # Fallback to defaults
    return {
        "pad_left": 0.1003,
        "pad_right": 0.1468,
        "pad_top": 0.1379,
        "pad_bottom": 0.1441
    }


def compute_foot_box(keypoints, img_w, img_h, scale_x=1.0, scale_y=1.0, calib=None):
    """
    Compute scale-aware calibrated bounding box from the 14 foot keypoints ONLY.
    Explicitly excludes ankle_center (12) and shin_mid (15) to match GT foot definition.
    
    Args:
        keypoints: list of [x, y, conf] for 16 keypoints (in heatmap or pixel coords)
        img_w, img_h: original image dimensions in pixels
        scale_x, scale_y: multiplier to convert keypoints to pixel coords
        calib: dict of padding ratios (pad_left, pad_right, pad_top, pad_bottom)
        
    Returns:
        (x1, y1, x2, y2) bounding box in pixel coordinates, or None if insufficient points
    """
    if calib is None:
        calib = load_box_calibration()

    if len(keypoints) == 13:
        FOOT_KP_INDICES = list(range(13))
    else:
        FOOT_KP_INDICES = [i for i in range(16) if i not in (12, 15)]
    valid_pts = []
    for idx in FOOT_KP_INDICES:
        if idx < len(keypoints):
            kp = keypoints[idx]
            if len(kp) >= 3 and kp[2] > 0.15:
                px = kp[0] * scale_x
                py = kp[1] * scale_y
                valid_pts.append((px, py))

    if len(valid_pts) < 4:
        return None

    xs = [p[0] for p in valid_pts]
    ys = [p[1] for p in valid_pts]
    k_min_x, k_max_x = min(xs), max(xs)
    k_min_y, k_max_y = min(ys), max(ys)
    kw = max(1.0, k_max_x - k_min_x)
    kh = max(1.0, k_max_y - k_min_y)

    pad_l = calib.get("pad_left", 0.10) * kw
    pad_r = calib.get("pad_right", 0.15) * kw
    pad_t = calib.get("pad_top", 0.14) * kh
    pad_b = calib.get("pad_bottom", 0.14) * kh

    x1 = int(max(0, round(k_min_x - pad_l)))
    y1 = int(max(0, round(k_min_y - pad_t)))
    x2 = int(min(img_w - 1, round(k_max_x + pad_r)))
    y2 = int(min(img_h - 1, round(k_max_y + pad_b)))

    return (x1, y1, x2, y2)


def decode_spatial_instances_single(
    heatmaps,
    class_probs,
    pafs=None,
    peak_threshold=0.10,
    nms_window=3,
    min_keypoints=5,
    max_instances=2,
    subpixel=True,
    max_radius=20.0,
    w_conf=1.0,
    w_paf=0.5,
    w_geom=0.20,
    max_iter=3,
    use_skeleton=True,
    paf_config=None
):
    """
    Instance decoding using Spatial Clustering of Heatmap Peaks with Skeleton Consistency:
    1. Discovers spatial instance centroids and Voronoi partition.
    2. Gathers top-3 candidate peaks per channel in each foot partition.
    3. Optimizes joint keypoint selection using PAF vector fields and learned pairwise distance priors.
    4. Prevents shrink bias and cross-foot limb jumps while preserving all keypoints (13 or 16).
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    if isinstance(class_probs, torch.Tensor):
        class_probs = class_probs.detach().cpu().numpy()
    if isinstance(pafs, torch.Tensor):
        pafs = pafs.detach().cpu().numpy()

    H, W = heatmaps.shape[1], heatmaps.shape[2]

    num_k = heatmaps.shape[0]

    # 1. Detect peaks across all channels
    all_peaks = {}
    for kp_idx in range(num_k):
        all_peaks[kp_idx] = detect_peaks(heatmaps[kp_idx], threshold=peak_threshold, nms_window=nms_window, subpixel=subpixel)

    # 2. Collect high-confidence anchor peaks to discover foot centroids:
    anchor_indices = [5, 10, 1, 3] if num_k == 13 else [5, 6, 12, 1]
    anchor_pts = []
    for kp_idx in anchor_indices:
        for p in all_peaks.get(kp_idx, []):
            if p[2] > 0.35:
                anchor_pts.append((p[0], p[1], kp_idx, p[2]))

    # Fallback to any strong foot keypoints if anchors sparse (exclude shin_mid kp 15 if 16-KP)
    if len(anchor_pts) < 4:
        for kp_idx in range(num_k):
            if num_k == 16 and kp_idx == 15:
                continue
            for p in all_peaks.get(kp_idx, []):
                if p[2] > 0.35:
                    anchor_pts.append((p[0], p[1], kp_idx, p[2]))

    if len(anchor_pts) < 4:
        for kp_idx in range(num_k):
            if num_k == 16 and kp_idx == 15:
                continue
            for p in all_peaks.get(kp_idx, []):
                if p[2] > 0.15:
                    anchor_pts.append((p[0], p[1], kp_idx, p[2]))

    if len(anchor_pts) < 3:
        return []

    # 3. Determine number of instances (1 or 2) using K-Means clustering
    pts_xy = np.array([[p[0], p[1]] for p in anchor_pts], dtype=np.float32)
    
    if max_instances >= 2 and len(pts_xy) >= 6:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
        _, labels, centers = cv2.kmeans(pts_xy, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        dist_between_centers = np.linalg.norm(centers[0] - centers[1])
        c0_count = np.sum(labels == 0)
        c1_count = np.sum(labels == 1)
        
        if dist_between_centers < 7.0 or c0_count < 3 or c1_count < 3:
            centers = [np.mean(pts_xy, axis=0)]
        else:
            centers = sorted(centers, key=lambda c: c[0])
    else:
        centers = [np.mean(pts_xy, axis=0)]

    PERIPHERAL_KPS = {0, 1, 2, 8, 9, 10} if num_k == 13 else {0, 1, 2, 9, 10, 11}
    LEG_KPS = set() if num_k == 13 else {12, 15}

    # Precompute Voronoi masks for partition fallback
    yy, xx = np.indices((H, W), dtype=np.float32)
    voronoi_masks = []
    for i, (cx, cy) in enumerate(centers):
        dist_this = (xx - cx)**2 + (yy - cy)**2
        mask = np.ones((H, W), dtype=bool)
        for j, (ocx, ocy) in enumerate(centers):
            if i != j:
                dist_other = (xx - ocx)**2 + (yy - ocy)**2
                mask &= (dist_this < dist_other)
        voronoi_masks.append(mask)

    # Skeleton graph setup
    if paf_config is None:
        paf_cfg = load_paf_config("paf_connections_13kp.yaml" if num_k == 13 else "paf_connections.yaml")
    else:
        paf_cfg = paf_config
    limb_pairs = paf_cfg.get("limb_pairs", [])

    skeleton_priors = load_skeleton_priors("skeleton_priors_train_13kp.json" if num_k == 13 else "skeleton_priors_train.json")

    neighbors = defaultdict(list)
    for l_idx, (u, v, _) in enumerate(limb_pairs):
        neighbors[u].append((v, l_idx, True))
        neighbors[v].append((u, l_idx, False))

    extra_edges = [(2, 5), (0, 5), (10, 5)] if num_k == 13 else [(2, 5), (0, 5), (11, 5)]
    for u, v in extra_edges:
        neighbors[u].append((v, -1, True))
        neighbors[v].append((u, -1, False))

    instances = []
    for c_idx, (cx, cy) in enumerate(centers):
        other_centers = [other for other in centers if not (np.isclose(other[0], cx) and np.isclose(other[1], cy))]
        foot_mask = voronoi_masks[c_idx]

        # Gather Top-3 candidates per keypoint in this foot's Voronoi region
        candidates_per_kp = {}
        for kp_idx in range(num_k):
            c_peaks = all_peaks.get(kp_idx, [])
            valid_cands = []
            max_r = 22.0 if (num_k == 16 and kp_idx == 15) else max_radius

            for p in c_peaks:
                dist = np.sqrt((p[0] - cx)**2 + (p[1] - cy)**2)
                if other_centers:
                    dist_other = min(np.sqrt((p[0] - oc[0])**2 + (p[1] - oc[1])**2) for oc in other_centers)
                    if dist >= dist_other:
                        continue
                if dist <= max_r and p[2] >= peak_threshold:
                    valid_cands.append(p)

            valid_cands.sort(key=lambda item: item[2], reverse=True)
            valid_cands = valid_cands[:3]

            if not valid_cands:
                # Voronoi partition fallback
                fallback_peak = None
                best_fb_conf = 0.0
                max_fb_r = 25.0 if kp_idx == 15 else 22.0
                for p in c_peaks:
                    dist = np.sqrt((p[0] - cx)**2 + (p[1] - cy)**2)
                    if other_centers:
                        dist_other = min(np.sqrt((p[0] - oc[0])**2 + (p[1] - oc[1])**2) for oc in other_centers)
                        if dist >= dist_other:
                            continue
                    if dist <= max_fb_r and p[2] > best_fb_conf:
                        best_fb_conf = p[2]
                        fallback_peak = p

                if fallback_peak is not None and best_fb_conf >= 0.08:
                    valid_cands.append(fallback_peak)
                else:
                    dist_grid_sq = (xx - cx)**2 + (yy - cy)**2
                    foot_bounded_mask = foot_mask & (dist_grid_sq <= max_fb_r**2)
                    masked_patch = np.where(foot_bounded_mask, heatmaps[kp_idx], 0.0)
                    local_max = float(masked_patch.max()) if masked_patch.size > 0 else 0.0
                    if local_max > 0.12:
                        py_l, px_l = np.unravel_index(np.argmax(masked_patch), masked_patch.shape)
                        valid_cands.append((float(px_l), float(py_l), local_max))

            candidates_per_kp[kp_idx] = valid_cands

        # Estimate instance scale
        scale_anchors = [5, 0, 10, 2] if num_k == 13 else [5, 6, 0, 11, 2]
        anchor_coords = [candidates_per_kp[k][0][:2] for k in scale_anchors if candidates_per_kp[k]]
        if len(anchor_coords) >= 2:
            pts_arr = np.array(anchor_coords)
            foot_scale = float(max(10.0, np.hypot(pts_arr[:, 0].max() - pts_arr[:, 0].min(),
                                                 pts_arr[:, 1].max() - pts_arr[:, 1].min()) * 1.3))
        else:
            foot_scale = 22.0

        current_selection = {k: (0 if candidates_per_kp[k] else -1) for k in range(num_k)}

        # Joint Skeleton Optimization via Iterated Conditional Modes (ICM)
        if pafs is not None and use_skeleton:
            for _ in range(max_iter):
                changed = False
                for k in range(num_k):
                    cands = candidates_per_kp[k]
                    if len(cands) <= 1:
                        continue

                    best_idx = current_selection[k]
                    best_score = -1e9

                    for cand_idx, p_cand in enumerate(cands):
                        conf = p_cand[2]
                        dist_to_c = np.hypot(p_cand[0] - cx, p_cand[1] - cy)
                        if k in PERIPHERAL_KPS:
                            unary = w_conf * conf
                        elif k in LEG_KPS:
                            unary = w_conf * conf - 0.020 * dist_to_c
                        else:
                            unary = w_conf * conf - 0.015 * dist_to_c

                        pairwise_score = 0.0
                        for nbr, limb_idx, is_from in neighbors[k]:
                            nbr_sel_idx = current_selection[nbr]
                            if nbr_sel_idx < 0 or not candidates_per_kp[nbr]:
                                continue
                            p_nbr = candidates_per_kp[nbr][nbr_sel_idx]
                            if p_nbr[2] < 0.30:
                                continue

                            # PAF alignment along limb
                            paf_score = 0.0
                            if limb_idx >= 0:
                                paf_x = pafs[limb_idx * 2]
                                paf_y = pafs[limb_idx * 2 + 1]
                                pt1 = (p_cand[0], p_cand[1]) if is_from else (p_nbr[0], p_nbr[1])
                                pt2 = (p_nbr[0], p_nbr[1]) if is_from else (p_cand[0], p_cand[1])
                                paf_val = compute_paf_score(paf_x, paf_y, pt1, pt2, num_samples=6)
                                paf_score = w_paf * paf_val

                            # Pairwise distance consistency
                            geom_pen = 0.0
                            prior_key = f"{k}_{nbr}" if is_from else f"{nbr}_{k}"
                            if prior_key in skeleton_priors:
                                prior = skeleton_priors[prior_key]
                                mu_d = prior["mean_dist"]
                                std_d = max(0.08, prior["std_dist"])
                                actual_dist = np.hypot(p_cand[0] - p_nbr[0], p_cand[1] - p_nbr[1])
                                actual_norm = actual_dist / foot_scale
                                z_score = abs(actual_norm - mu_d) / std_d
                                if z_score > 1.5:
                                    geom_pen = min(2.0, ((z_score - 1.5) ** 2) * 0.5)

                            eff_w_geom = 0.05 if k in (1, 10) else w_geom
                            nbr_weight = min(1.0, p_nbr[2])
                            pairwise_score += nbr_weight * (paf_score - eff_w_geom * geom_pen)

                        total_cand_score = unary + pairwise_score
                        if total_cand_score > best_score:
                            best_score = total_cand_score
                            best_idx = cand_idx

                    if best_idx != current_selection[k]:
                        current_selection[k] = best_idx
                        changed = True

                if not changed:
                    break

        inst_kps = []
        scores = []
        for k in range(num_k):
            sel_idx = current_selection[k]
            if sel_idx >= 0 and candidates_per_kp[k]:
                p = candidates_per_kp[k][sel_idx]
                inst_kps.append([float(p[0]), float(p[1]), float(p[2])])
                scores.append(float(p[2]))
            else:
                inst_kps.append([0.0, 0.0, 0.0])
                
        # Count detected keypoints
        valid_kps = [k for k in inst_kps if k[2] > 0.15]
        if len(valid_kps) < min_keypoints:
            continue
            
        # Class assignment: sample class probability map at valid keypoint positions AND around foot center
        left_votes = []
        right_votes = []
        for kx, ky, kv in valid_kps:
            px, py = int(np.clip(round(kx), 0, W - 1)), int(np.clip(round(ky), 0, H - 1))
            left_votes.append(class_probs[0, py, px])
            right_votes.append(class_probs[1, py, px])

        # Also sample a 7x7 patch around the cluster center for spatial continuity
        icx, icy = int(np.clip(round(cx), 0, W - 1)), int(np.clip(round(cy), 0, H - 1))
        r_c = 4
        center_patch_left = class_probs[0, max(0, icy-r_c):min(H, icy+r_c+1), max(0, icx-r_c):min(W, icx+r_c+1)]
        center_patch_right = class_probs[1, max(0, icy-r_c):min(H, icy+r_c+1), max(0, icx-r_c):min(W, icx+r_c+1)]
        if center_patch_left.size > 0:
            left_votes.append(float(np.mean(center_patch_left)))
            right_votes.append(float(np.mean(center_patch_right)))

        mean_left = np.mean(left_votes) if left_votes else 0.5
        mean_right = np.mean(right_votes) if right_votes else 0.5

        pred_cls = 0 if mean_left > mean_right else 1
        conf = float(max(mean_left, mean_right))

        instances.append({
            'class_id': pred_cls,
            'class_confidence': conf,
            'keypoints': inst_kps,
            'score': float(np.mean(scores)) if scores else 0.0,
            'center': (float(cx), float(cy)),
            'margin': float(mean_left - mean_right)
        })
        
    # If 2 instances got assigned the same class, disambiguate using margin
    if len(instances) == 2 and instances[0]['class_id'] == instances[1]['class_id']:
        if instances[0]['margin'] > instances[1]['margin']:
            instances[0]['class_id'] = 0
            instances[1]['class_id'] = 1
        else:
            instances[0]['class_id'] = 1
            instances[1]['class_id'] = 0

    # Sort by score descending
    instances = sorted(instances, key=lambda i: i['score'], reverse=True)
    return instances[:max_instances]



def decode_paf_instances_single(
    heatmaps,
    pafs,
    class_probs,
    peak_threshold=0.10,
    nms_window=3,
    max_peaks_per_channel=10,
    paf_threshold=0.05,
    max_limb_dist=28.0,
    min_keypoints=4,
    max_instances=2,
    subpixel=True,
    paf_config=None
):
    """
    OpenPose-inspired PAF-led multi-instance keypoint decoder.
    
    1. Plateau-safe peak detection with subpixel quadratic Taylor expansion refinement.
    2. Line integral PAF candidate scoring along 16 limbs with distance bounds.
    3. One-to-one disjoint bipartite matching per limb.
    4. Graph tree instance assembly strictly enforcing one peak per keypoint type,
       with contradiction rejection for conflicting merges.
    5. Post-assembly left/right foot assignment from class head probabilities.
    
    Args:
        heatmaps: [K, H, W] (K=13 or 16)
        pafs: [2*L, H, W] (32 channels for 16 limbs)
        class_probs: [2, H, W]
        peak_threshold: Minimum peak confidence
        nms_window: Window size for NMS
        max_peaks_per_channel: Max peaks per keypoint channel
        paf_threshold: Minimum PAF line integral score
        max_limb_dist: Maximum physical pixel distance between keypoints on 64x64 grid
        min_keypoints: Minimum keypoints required to form a valid foot instance
        max_instances: Maximum instances to return (default 2)
        subpixel: Apply quadratic subpixel refinement
        paf_config: PAF config dict or path
        
    Returns:
        instances: List of dicts with 'keypoints' (64x64), 'score', 'class_id', 'class_confidence'
    """
    from scipy.ndimage import maximum_filter, label as nd_label

    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    if isinstance(pafs, torch.Tensor):
        pafs = pafs.detach().cpu().numpy()
    if isinstance(class_probs, torch.Tensor):
        class_probs = class_probs.detach().cpu().numpy()

    num_k = heatmaps.shape[0]
    H, W = heatmaps.shape[1], heatmaps.shape[2]

    # Load PAF config
    if paf_config is None:
        paf_cfg = load_paf_config("paf_connections_13kp.yaml" if num_k == 13 else "paf_connections.yaml")
    elif isinstance(paf_config, (str, Path)):
        with open(paf_config, 'r', encoding='utf-8') as f:
            paf_cfg = yaml.safe_load(f)
    else:
        paf_cfg = paf_config
    limb_pairs = paf_cfg.get('limb_pairs', [])
    num_limbs = len(limb_pairs)

    # -------------------------------------------------------------------------
    # 1. Plateau-Safe Peak Detection with Subpixel Refinement
    # -------------------------------------------------------------------------
    all_peaks = {}
    for k in range(num_k):
        hm = heatmaps[k]
        local_max = maximum_filter(hm, size=nms_window)
        peaks_binary = (hm == local_max) & (hm >= peak_threshold)
        
        labeled_mask, num_features = nd_label(peaks_binary)
        k_peaks = []
        if num_features > 0:
            for feat_idx in range(1, num_features + 1):
                ys, xs = np.where(labeled_mask == feat_idx)
                best_i = np.argmax(hm[ys, xs])
                y, x = int(ys[best_i]), int(xs[best_i])
                conf = float(hm[y, x])
                
                sub_x, sub_y = float(x), float(y)
                if subpixel and 1 <= x < W - 1 and 1 <= y < H - 1:
                    dx = 0.5 * (hm[y, x + 1] - hm[y, x - 1])
                    dxx = hm[y, x + 1] - 2.0 * hm[y, x] + hm[y, x - 1]
                    if abs(dxx) > 1e-5:
                        offset_x = -dx / dxx
                        sub_x += float(np.clip(offset_x, -0.5, 0.5))
                    dy = 0.5 * (hm[y + 1, x] - hm[y - 1, x])
                    dyy = hm[y + 1, x] - 2.0 * hm[y, x] + hm[y - 1, x]
                    if abs(dyy) > 1e-5:
                        offset_y = -dy / dyy
                        sub_y += float(np.clip(offset_y, -0.5, 0.5))
                k_peaks.append((sub_x, sub_y, conf))
                
        k_peaks.sort(key=lambda p: p[2], reverse=True)
        all_peaks[k] = k_peaks[:max_peaks_per_channel]

    # -------------------------------------------------------------------------
    # 2. Score Candidate Pairs along Configured Limbs using PAF Line Integrals
    # -------------------------------------------------------------------------
    all_matched_connections = []

    for limb_idx, (u, v, name) in enumerate(limb_pairs):
        if limb_idx * 2 + 1 >= pafs.shape[0] or u >= num_k or v >= num_k:
            continue
            
        peaks_u = all_peaks.get(u, [])
        peaks_v = all_peaks.get(v, [])
        if not peaks_u or not peaks_v:
            continue

        paf_x = pafs[limb_idx * 2]
        paf_y = pafs[limb_idx * 2 + 1]

        limb_connections = []
        for i, p_u in enumerate(peaks_u):
            for j, p_v in enumerate(peaks_v):
                dx = p_v[0] - p_u[0]
                dy = p_v[1] - p_u[1]
                dist = np.hypot(dx, dy)
                if dist > max_limb_dist or dist < 0.5:
                    continue

                ux = dx / dist
                uy = dy / dist

                # Sample 10 points along the line segment
                t_vals = np.linspace(0.0, 1.0, 10, dtype=np.float32)
                sample_xs = np.clip(np.round(p_u[0] + t_vals * dx).astype(int), 0, W - 1)
                sample_ys = np.clip(np.round(p_u[1] + t_vals * dy).astype(int), 0, H - 1)

                dot_prods = paf_x[sample_ys, sample_xs] * ux + paf_y[sample_ys, sample_xs] * uy
                valid_count = int(np.sum(dot_prods > 0.05))
                line_score = float(np.mean(dot_prods))

                # OpenPose criterion: at least 40% positive samples and positive mean integral
                if valid_count >= 4 and line_score > paf_threshold:
                    comb_score = line_score + 0.5 * (p_u[2] + p_v[2])
                    limb_connections.append((comb_score, i, j, p_u, p_v))

        # -------------------------------------------------------------------------
        # 3. Disjoint One-to-One Bipartite Matching for this Limb
        # -------------------------------------------------------------------------
        limb_connections.sort(key=lambda item: item[0], reverse=True)
        matched_u = set()
        matched_v = set()
        for comb_score, i, j, p_u, p_v in limb_connections:
            if i not in matched_u and j not in matched_v:
                matched_u.add(i)
                matched_v.add(j)
                all_matched_connections.append({
                    'score': comb_score,
                    'limb_idx': limb_idx,
                    'u': u,
                    'v': v,
                    'peak_u_idx': i,
                    'peak_v_idx': j,
                    'p_u': p_u,
                    'p_v': p_v
                })

    # Sort all limb matches across the entire skeleton by affinity score descending
    all_matched_connections.sort(key=lambda item: item['score'], reverse=True)

    # -------------------------------------------------------------------------
    # 4. Instance Assembly with Strict Contradiction Rejection
    # -------------------------------------------------------------------------
    # instances: list of dicts: {'kps': {kp_type: (x, y, conf, cand_idx)}, 'score': float, 'edges': int}
    assembled_instances = []

    for conn in all_matched_connections:
        u = conn['u']
        v = conn['v']
        cand_u = conn['peak_u_idx']
        cand_v = conn['peak_v_idx']
        p_u = conn['p_u']
        p_v = conn['p_v']
        score = conn['score']

        inst_u_idx = None
        inst_v_idx = None

        for idx, inst in enumerate(assembled_instances):
            if u in inst['kps'] and inst['kps'][u][3] == cand_u:
                inst_u_idx = idx
            if v in inst['kps'] and inst['kps'][v][3] == cand_v:
                inst_v_idx = idx

        if inst_u_idx is None and inst_v_idx is None:
            # Create a new instance
            new_inst = {
                'kps': {
                    u: (p_u[0], p_u[1], p_u[2], cand_u),
                    v: (p_v[0], p_v[1], p_v[2], cand_v)
                },
                'score': score,
                'edges': 1
            }
            assembled_instances.append(new_inst)

        elif inst_u_idx is not None and inst_v_idx is None:
            # Add v to inst_u if no conflicting v exists
            inst = assembled_instances[inst_u_idx]
            if v not in inst['kps']:
                inst['kps'][v] = (p_v[0], p_v[1], p_v[2], cand_v)
                inst['score'] += score
                inst['edges'] += 1
            # If v already in inst with different cand_v, reject (contradiction)

        elif inst_u_idx is None and inst_v_idx is not None:
            # Add u to inst_v if no conflicting u exists
            inst = assembled_instances[inst_v_idx]
            if u not in inst['kps']:
                inst['kps'][u] = (p_u[0], p_u[1], p_u[2], cand_u)
                inst['score'] += score
                inst['edges'] += 1

        elif inst_u_idx is not None and inst_v_idx is not None:
            if inst_u_idx == inst_v_idx:
                # Same instance (cycle/loop completion)
                inst = assembled_instances[inst_u_idx]
                inst['score'] += score
                inst['edges'] += 1
            else:
                # Attempting to merge two instances: CONTRADICTION CHECK
                inst_u = assembled_instances[inst_u_idx]
                inst_v = assembled_instances[inst_v_idx]
                common_types = set(inst_u['kps'].keys()) & set(inst_v['kps'].keys())
                has_contradiction = False
                for c_k in common_types:
                    if inst_u['kps'][c_k][3] != inst_v['kps'][c_k][3]:
                        has_contradiction = True
                        break

                if not has_contradiction:
                    # Merge inst_v into inst_u
                    for k_idx, val in inst_v['kps'].items():
                        inst_u['kps'][k_idx] = val
                    inst_u['score'] += inst_v['score'] + score
                    inst_u['edges'] += inst_v['edges'] + 1
                    assembled_instances.pop(inst_v_idx)
                # Else: Contradictory merge rejected! Keep as separate instances.

    # -------------------------------------------------------------------------
    # 5. Format Instances & Filter Spurious Fragments
    # -------------------------------------------------------------------------
    valid_instances = []
    for inst in assembled_instances:
        if len(inst['kps']) < min_keypoints:
            continue

        kps_list = []
        conf_list = []
        for k in range(num_k):
            if k in inst['kps']:
                x, y, conf, _ = inst['kps'][k]
                kps_list.append([x, y, conf])
                conf_list.append(conf)
            else:
                kps_list.append([0.0, 0.0, 0.0])

        avg_conf = float(np.mean(conf_list)) if conf_list else 0.0
        inst_score = avg_conf + (inst['score'] / max(1, inst['edges']))
        valid_instances.append({
            'keypoints': kps_list,
            'score': inst_score,
            'num_keypoints': len(inst['kps'])
        })

    valid_instances.sort(key=lambda item: item['score'], reverse=True)
    valid_instances = valid_instances[:max_instances]

    # -------------------------------------------------------------------------
    # 6. Assign Left / Right Class using Class Map
    # -------------------------------------------------------------------------
    for inst in valid_instances:
        left_votes = []
        right_votes = []
        for kx, ky, kv in inst['keypoints']:
            if kv > 0:
                ix = int(np.clip(np.round(kx), 0, W - 1))
                iy = int(np.clip(np.round(ky), 0, H - 1))
                left_votes.append(float(class_probs[0, iy, ix]))
                right_votes.append(float(class_probs[1, iy, ix]))

        p_left = float(np.mean(left_votes)) if left_votes else 0.5
        p_right = float(np.mean(right_votes)) if right_votes else 0.5
        inst['p_left'] = p_left
        inst['p_right'] = p_right

    if len(valid_instances) == 1:
        if valid_instances[0]['p_left'] >= valid_instances[0]['p_right']:
            valid_instances[0]['class_id'] = 0
            valid_instances[0]['class_name'] = 'left_foot'
            valid_instances[0]['class_confidence'] = valid_instances[0]['p_left']
        else:
            valid_instances[0]['class_id'] = 1
            valid_instances[0]['class_name'] = 'right_foot'
            valid_instances[0]['class_confidence'] = valid_instances[0]['p_right']
    elif len(valid_instances) >= 2:
        # Resolve joint assignment: L0+R1 vs R0+L1
        score_lr = valid_instances[0]['p_left'] + valid_instances[1]['p_right']
        score_rl = valid_instances[0]['p_right'] + valid_instances[1]['p_left']
        if score_lr >= score_rl:
            valid_instances[0]['class_id'] = 0
            valid_instances[0]['class_name'] = 'left_foot'
            valid_instances[0]['class_confidence'] = valid_instances[0]['p_left']
            valid_instances[1]['class_id'] = 1
            valid_instances[1]['class_name'] = 'right_foot'
            valid_instances[1]['class_confidence'] = valid_instances[1]['p_right']
        else:
            valid_instances[0]['class_id'] = 1
            valid_instances[0]['class_name'] = 'right_foot'
            valid_instances[0]['class_confidence'] = valid_instances[0]['p_right']
            valid_instances[1]['class_id'] = 0
            valid_instances[1]['class_name'] = 'left_foot'
            valid_instances[1]['class_confidence'] = valid_instances[1]['p_left']

    return valid_instances


def group_keypoints(heatmaps, pafs, class_probs, 
                    peak_threshold=0.1, 
                    paf_threshold=0.05, 
                    nms_window=3,
                    min_keypoints=5,
                    max_instances=2,
                    subpixel=True,
                    method='spatial',
                    paf_config=None):
    """
    Complete keypoint grouping pipeline
    
    Args:
        heatmaps: [B, K, H, W] keypoint heatmaps (K=13 or 16)
        pafs: [B, 2*L, H, W] PAF fields
        class_probs: [B, 2, H, W] class probability maps
        peak_threshold: Minimum peak confidence
        paf_threshold: Minimum PAF score (for method='paf')
        nms_window: NMS window size
        min_keypoints: Minimum keypoints per instance
        max_instances: Maximum instances to return per image
        subpixel: Whether to apply quadratic sub-pixel peak refinement
        method: 'spatial' (default, robust centroid cluster decoding) or 'paf' (legacy)
        paf_config: Optional custom PAF config dictionary
    
    Returns:
        batch_instances: List of length B, each containing list of instances
    """
    batch_size = heatmaps.shape[0]
    num_k = heatmaps.shape[1]
    batch_instances = []

    if method == 'paf_assembly':
        for b in range(batch_size):
            paf_b = pafs[b] if pafs is not None else None
            cls_b = class_probs[b] if class_probs is not None else None
            insts = decode_paf_instances_single(
                heatmaps=heatmaps[b],
                pafs=paf_b,
                class_probs=cls_b,
                peak_threshold=peak_threshold,
                nms_window=nms_window,
                min_keypoints=min_keypoints,
                max_instances=max_instances,
                subpixel=subpixel,
                paf_config=paf_config
            )
            batch_instances.append(insts)
        return batch_instances

    if method == 'spatial':
        for b in range(batch_size):
            paf_b = pafs[b] if pafs is not None else None
            insts = decode_spatial_instances_single(
                heatmaps=heatmaps[b],
                class_probs=class_probs[b],
                pafs=paf_b,
                peak_threshold=peak_threshold,
                nms_window=nms_window,
                min_keypoints=min_keypoints,
                max_instances=max_instances,
                subpixel=subpixel,
                paf_config=paf_config
            )
            batch_instances.append(insts)
        return batch_instances

    # Legacy PAF method
    if paf_config is None:
        config = load_paf_config("paf_connections_13kp.yaml" if num_k == 13 else "paf_connections.yaml")
    else:
        config = paf_config
    limb_pairs = config['limb_pairs']
    
    for b in range(batch_size):
        # Detect peaks
        all_peaks = {}
        for kp_idx in range(num_k):
            heatmap = heatmaps[b, kp_idx]
            peaks = detect_peaks(heatmap, peak_threshold, nms_window, subpixel=subpixel)
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
