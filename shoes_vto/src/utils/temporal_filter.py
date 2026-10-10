"""
OneEuro Temporal Filter for Keypoint Video Stabilization

Based on:
Casiez, G., Roussel, N., & Vogel, F. (2012).
1 € filter: a simple speed-based low-pass filter for noisy input in HCI.
Proceedings of the SIGCHI Conference on Human Factors in Computing Systems.

Eliminates jitter in static postures while preventing lag in fast movements.
"""

import math
import numpy as np


class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev = None

    def reset(self):
        self.prev = None

    def filter(self, val):
        if self.prev is None:
            self.prev = val
            return val
        out = self.alpha * val + (1.0 - self.alpha) * self.prev
        self.prev = out
        return out


class OneEuroFilter1D:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        """
        1D OneEuro Filter
        
        Args:
            min_cutoff: Minimum cutoff frequency (Hz). Lower = less jitter when static.
            beta: Speed coefficient. Higher = less lag during rapid movement.
            d_cutoff: Cutoff frequency for derivative (Hz).
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = max(1e-4, t - self.t_prev)
        self.t_prev = t

        # Estimate derivative (speed)
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Adaptive cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)

        # Filtered output
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat


class FootKeypointFilter:
    """
    Stabilizes 16 2D keypoints for a single foot instance across consecutive frames.
    """
    def __init__(self, min_cutoff=1.0, beta=0.008, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        # 16 keypoints * 2 coordinates (x, y)
        self.filters_x = [OneEuroFilter1D(min_cutoff, beta, d_cutoff) for _ in range(16)]
        self.filters_y = [OneEuroFilter1D(min_cutoff, beta, d_cutoff) for _ in range(16)]

    def reset(self):
        for f in self.filters_x:
            f.reset()
        for f in self.filters_y:
            f.reset()

    def filter(self, keypoints, timestamp):
        """
        Filter a list of 16 keypoints: [[x, y, conf], ...]
        
        Args:
            keypoints: List of 16 elements [x, y, conf]
            timestamp: Float timestamp in seconds (or frame index / fps)
            
        Returns:
            smoothed_keypoints: List of 16 elements [x_smooth, y_smooth, conf]
        """
        smoothed = []
        for i in range(16):
            if i < len(keypoints):
                x, y, conf = keypoints[i][:3]
                if conf > 0.1:  # Only smooth if keypoint is detected with reasonable confidence
                    x_s = self.filters_x[i].filter(float(x), timestamp)
                    y_s = self.filters_y[i].filter(float(y), timestamp)
                    smoothed.append([x_s, y_s, conf])
                else:
                    smoothed.append([x, y, conf])
            else:
                smoothed.append([0.0, 0.0, 0.0])
        return smoothed


class MultiFootTracker:
    """
    Tracks and smooths multiple feet (left and right) across video frames.
    Maintains persistent IDs based on Left/Right classification and spatial proximity.
    """
    def __init__(self, min_cutoff=1.0, beta=0.008):
        self.feet = {
            0: FootKeypointFilter(min_cutoff, beta),  # Left foot (class 0)
            1: FootKeypointFilter(min_cutoff, beta)   # Right foot (class 1)
        }
        self.last_seen = {0: -1, 1: -1}

    def reset(self):
        for f in self.feet.values():
            f.reset()
        self.last_seen = {0: -1, 1: -1}

    def update(self, instances, timestamp):
        """
        Update tracker with detected instances in current frame.
        
        Args:
            instances: List of instance dicts [{'keypoints': ..., 'class_id': 0 or 1, 'score': ...}, ...]
            timestamp: Current timestamp (seconds)
            
        Returns:
            smoothed_instances: List of instance dicts with smoothed keypoints
        """
        smoothed_instances = []
        for inst in instances:
            class_id = inst.get('class_id', 0)
            kps = inst.get('keypoints', [])
            
            if class_id in self.feet:
                smoothed_kps = self.feet[class_id].filter(kps, timestamp)
                self.last_seen[class_id] = timestamp
            else:
                smoothed_kps = kps
                
            new_inst = dict(inst)
            new_inst['keypoints'] = smoothed_kps
            smoothed_instances.append(new_inst)
            
        return smoothed_instances

