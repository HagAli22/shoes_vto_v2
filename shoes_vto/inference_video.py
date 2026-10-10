"""
Real-Time Video Inference & Evaluation Pipeline for ARShoe M1v2

Runs keypoint estimation, PAF limb association, sub-pixel refinement,
and OneEuro temporal stabilization on video files or webcam streams.
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.arshoe_m1v2 import ARShoeM1v2
from utils.keypoint_grouping import group_keypoints, load_paf_config, compute_foot_box
from utils.temporal_filter import MultiFootTracker

# Colors (BGR)
COLOR_LEFT = (235, 130, 30)      # Sky Blue / Cyan for Left Foot
COLOR_RIGHT = (40, 60, 240)      # Vivid Red/Orange for Right Foot
COLOR_KP_LEFT = (255, 200, 50)   # Highlight Cyan
COLOR_KP_RIGHT = (80, 100, 255)  # Highlight Orange/Red
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (20, 20, 20)

KEYPOINT_NAMES = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial", "ball_lateral",
    "ball_top", "instep_top", "arch_medial", "midfoot_lateral",
    "malleolus_medial", "malleolus_lateral", "toe_tip", "ankle_center",
    "throat", "achilles", "shin_mid"
]


def load_pytorch_model(checkpoint_path, device='cuda'):
    print(f"Loading PyTorch checkpoint from: {checkpoint_path}")
    model = ARShoeM1v2()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt
    cleaned_sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(cleaned_sd)
    model.to(device)
    model.eval()
    return model


def load_onnx_model(onnx_path):
    import onnxruntime as ort
    print(f"Loading ONNX model from: {onnx_path}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return session


def draw_hud(img, fps, inf_time_ms, post_time_ms, device_str, frame_idx, total_frames):
    """Draw telemetry Heads-Up Display in top-left corner."""
    hud_h, hud_w = 120, 420
    overlay = img.copy()
    cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (15, 15), (15 + hud_w, 15 + hud_h), (80, 80, 80), 1)

    # Title
    cv2.putText(img, "ARShoe M1v2 Real-Time Tracker", (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
    
    # Telemetry lines
    t_text = f"FPS: {fps:.1f} | Latency: {inf_time_ms + post_time_ms:.1f} ms"
    d_text = f"Inference: {inf_time_ms:.1f} ms | Postproc: {post_time_ms:.1f} ms ({device_str})"
    f_text = f"Frame: {frame_idx + 1}/{total_frames if total_frames > 0 else 'Live'}"
    
    cv2.putText(img, t_text, (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, d_text, (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, f_text, (25, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_instance(img, instance, limb_pairs, orig_w, orig_h, scale_x, scale_y):
    """Draw foot skeleton, keypoints, and classification tag."""
    kps = instance['keypoints']
    class_id = instance.get('class_id', 0)
    conf = instance.get('class_confidence', instance.get('score', 0.9))
    side_str = "Left Foot" if class_id == 0 else "Right Foot"
    color_main = COLOR_LEFT if class_id == 0 else COLOR_RIGHT
    color_kp = COLOR_KP_LEFT if class_id == 0 else COLOR_KP_RIGHT

    # 1. Scale keypoints to original video dimensions
    pts = []
    visible_pts = []
    for kp in kps:
        x, y, c = kp[:3]
        px = int(x * scale_x)
        py = int(y * scale_y)
        pts.append((px, py, c))
        if c > 0.15:
            visible_pts.append((px, py))

    # 2. Draw PAF skeleton limbs
    for from_idx, to_idx, _ in limb_pairs:
        if from_idx < len(pts) and to_idx < len(pts):
            p1 = pts[from_idx]
            p2 = pts[to_idx]
            if p1[2] > 0.15 and p2[2] > 0.15:
                cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color_main, 3, cv2.LINE_AA)

    # 3. Draw Keypoints
    for idx, (px, py, c) in enumerate(pts):
        if c > 0.15:
            cv2.circle(img, (px, py), 5, color_kp, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 6, (20, 20, 20), 1, cv2.LINE_AA)

    # 4. Draw bounding box and side badge
    box = compute_foot_box(pts, orig_w, orig_h, scale_x=1.0, scale_y=1.0)
    if box is not None:
        min_x, min_y, max_x, max_y = box

        # Bounding box
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), color_main, 2, cv2.LINE_AA)

        # Tag badge
        badge_text = f"{side_str}: {conf * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        badge_y1 = max(0, min_y - th - 12)
        badge_y2 = min_y
        cv2.rectangle(img, (min_x, badge_y1), (min_x + tw + 14, badge_y2), color_main, -1)
        cv2.putText(img, badge_text, (min_x + 7, badge_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def run_video_inference(
    video_path,
    output_path,
    checkpoint_path=None,
    onnx_path=None,
    device='cuda',
    smooth=True,
    subpixel=True,
    max_frames=None,
    show=False
):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("=" * 80)
    print("🎥 ARShoe M1v2 Real-Time Video Processor")
    print("=" * 80)
    print(f"  Input Video  : {video_path}")
    print(f"  Resolution   : {orig_w}x{orig_h} @ {orig_fps:.2f} FPS")
    print(f"  Total Frames : {total_frames}")
    print(f"  Temporal Smooth (OneEuro): {smooth}")
    print(f"  Sub-Pixel Refinement     : {subpixel}")

    # Setup model
    is_onnx = onnx_path is not None
    if is_onnx:
        session = load_onnx_model(onnx_path)
        input_name = session.get_inputs()[0].name
        device_str = "ONNX"
    else:
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        model = load_pytorch_model(checkpoint_path, device=device)
        device_str = f"PyTorch {device.upper()}"

    # Setup Video Writer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(output_path), fourcc, orig_fps, (orig_w, orig_h))

    # PAF and tracking configs
    paf_config = load_paf_config()
    limb_pairs = paf_config['limb_pairs']
    tracker = MultiFootTracker(min_cutoff=1.0, beta=0.008) if smooth else None

    # Processing loop
    scale_x = orig_w / 64.0  # Heatmap 64x64 to original resolution
    scale_y = orig_h / 64.0
    frame_idx = 0
    start_time = time.time()
    
    latencies_inf = []
    latencies_post = []

    print("\n🚀 Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        timestamp = frame_idx / orig_fps

        # 1. Preprocessing (Resize to 256x256, RGB, float [0, 1])
        t_pre = time.perf_counter()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        img_norm = img_resized.astype(np.float32) / 255.0

        # 2. Forward Inference
        t_inf_start = time.perf_counter()
        if is_onnx:
            input_tensor = np.expand_dims(np.transpose(img_norm, (2, 0, 1)), axis=0)
            outputs = session.run(None, {input_name: input_tensor})
            hm = torch.from_numpy(outputs[0])
            paf = torch.from_numpy(outputs[1])
            cls_probs = torch.from_numpy(outputs[2])
        else:
            input_tensor = torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_tensor)
                hm = out['heatmaps'].cpu()
                paf = out['pafs'].cpu()
                cls_probs = out['class'].cpu()
        t_inf_end = time.perf_counter()
        inf_ms = (t_inf_end - t_inf_start) * 1000.0
        latencies_inf.append(inf_ms)

        # 3. Postprocessing (Peak Detection, PAF Assembly, Class Voting)
        t_post_start = time.perf_counter()
        batch_instances = group_keypoints(
            heatmaps=hm,
            pafs=paf,
            class_probs=cls_probs,
            peak_threshold=0.12,
            paf_threshold=0.10,
            nms_window=3,
            min_keypoints=5,
            max_instances=2,
            subpixel=subpixel
        )
        raw_instances = batch_instances[0]
        # Keep instances with sufficient limb connectivity score
        instances = [inst for inst in raw_instances if inst.get('score', 0) >= 0.15]

        # 4. Temporal Smoothing via OneEuro Filter
        if tracker is not None:
            instances = tracker.update(instances, timestamp)
        t_post_end = time.perf_counter()
        post_ms = (t_post_end - t_post_start) * 1000.0
        latencies_post.append(post_ms)

        # 5. Visualization Rendering
        annotated_frame = frame.copy()
        for inst in instances:
            draw_instance(annotated_frame, inst, limb_pairs, orig_w, orig_h, scale_x, scale_y)

        # 6. Telemetry HUD
        fps_current = 1000.0 / max(1.0, inf_ms + post_ms)
        draw_hud(annotated_frame, fps_current, inf_ms, post_ms, device_str, frame_idx, total_frames)

        # Write frame
        out_writer.write(annotated_frame)

        if show:
            cv2.imshow("ARShoe M1v2 Real-Time Tracker", cv2.resize(annotated_frame, (720, 720)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"  Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) | "
                  f"Inf: {inf_ms:.1f} ms | Post: {post_ms:.1f} ms | FPS: {fps_current:.1f}")

    cap.release()
    out_writer.release()
    if show:
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_inf = np.mean(latencies_inf) if latencies_inf else 0.0
    avg_post = np.mean(latencies_post) if latencies_post else 0.0
    overall_fps = frame_idx / total_time if total_time > 0 else 0.0

    print("=" * 80)
    print("✅ Video Processing Complete!")
    print("=" * 80)
    print(f"  Frames Processed    : {frame_idx}")
    print(f"  Total Elapsed Time  : {total_time:.2f} s")
    print(f"  Average Inference   : {avg_inf:.2f} ms")
    print(f"  Average Postprocess : {avg_post:.2f} ms")
    print(f"  Average Total Latency: {avg_inf + avg_post:.2f} ms")
    print(f"  Effective Throughput: {overall_fps:.1f} FPS")
    print(f"  Annotated Video Saved: {output_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Video Inference with ARShoe M1v2")
    parser.add_argument('--input', type=str, default='outputs/test_video.mp4', help='Input video file or webcam index')
    parser.add_argument('--output', type=str, default='outputs/test_video_annotated.mp4', help='Output video file')
    parser.add_argument('--checkpoint', type=str, default='outputs/m1v2_training/checkpoints/best.pth', help='PyTorch checkpoint')
    parser.add_argument('--onnx', type=str, default=None, help='Optional ONNX model path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Inference device for PyTorch')
    parser.add_argument('--no_smooth', action='store_true', help='Disable OneEuro temporal filter')
    parser.add_argument('--no_subpixel', action='store_true', help='Disable sub-pixel peak refinement')
    parser.add_argument('--max_frames', type=int, default=None, help='Limit number of frames to process')
    parser.add_argument('--show', action='store_true', help='Display live preview window')
    args = parser.parse_args()

    run_video_inference(
        video_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx,
        device=args.device,
        smooth=not args.no_smooth,
        subpixel=not args.no_subpixel,
        max_frames=args.max_frames,
        show=args.show
    )


if __name__ == '__main__':
    main()

