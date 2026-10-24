"""
Patch trainer_m1.py to include live test visualizations every 10 epochs.
"""

from pathlib import Path

TRAINER_PATH = Path("shoes_vto/src/training/trainer_m1.py")


def patch_trainer():
    with open(TRAINER_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Add hook inside train loop
    old_loop = """            else:
                # Skip validation, just print train losses
                print(f"\\nEpoch {self.current_epoch} - Train: {train_losses['total']:.4f} "
                      f"(HM: {train_losses['heatmap']:.4f}, PAF: {train_losses['paf']:.4f}, "
                      f"Cls: {train_losses['class']:.4f})"
                      f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")"""

    new_loop = old_loop + """

            # Run test visualization at specified interval (default: every 10 epochs or epoch 1)
            vis_interval = self.config.get('vis_interval', 10)
            if vis_interval > 0 and (self.current_epoch % vis_interval == 0 or self.current_epoch == 1 or self.current_epoch == num_epochs):
                self.visualize_epoch(self.current_epoch)"""

    if new_loop not in content:
        if old_loop in content:
            content = content.replace(old_loop, new_loop, 1)
            print("Successfully added visualization hook to train() loop.")
        else:
            print("Warning: old_loop not found!")

    # 2. Add visualize_epoch method to ARShoeM1Trainer
    old_end = """        print("\\n" + "=" * 80)
        print(f"\u2705 Training complete! Best val loss: {self.best_val_loss:.4f}")
        print("=" * 80)"""

    vis_method = """        print("\\n" + "=" * 80)
        print(f"\u2705 Training complete! Best val loss: {self.best_val_loss:.4f}")
        print("=" * 80)

    def visualize_epoch(self, epoch):
        \"\"\"
        Generate and save side-by-side test visualizations during training.
        Allows inspecting model predictions live every 10 epochs.
        \"\"\"
        import cv2
        import numpy as np

        vis_dir = self.checkpoint_dir.parent / "epoch_visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        num_k = self.config.get('num_keypoints', 13)
        limbs = [
            (10, 0), (0, 2), (2, 1), (3, 4), (3, 5), (4, 5),
            (2, 6), (2, 7), (6, 5), (7, 5), (5, 11), (11, 8),
            (11, 9), (8, 12), (9, 12), (12, 1)
        ] if num_k == 13 else [
            (11, 0), (0, 2), (2, 1), (3, 4), (3, 6), (4, 6),
            (2, 7), (2, 8), (7, 6), (8, 6), (6, 13), (13, 9),
            (13, 10), (9, 14), (10, 14), (14, 1)
        ]

        kp_colors = [
            (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
            (255, 0, 0), (255, 0, 128), (128, 0, 255), (0, 165, 255), (147, 20, 255),
            (50, 205, 50), (208, 224, 64), (180, 105, 255)
        ]

        samples_to_vis = []
        n_samples = min(4, len(self.val_dataset))
        for i in range(n_samples):
            samples_to_vis.append(self.val_dataset[i])

        from utils.keypoint_grouping import decode_spatial_instances_single

        self.model.eval()
        panels = []
        panel_w, panel_h = 280, 280

        with torch.no_grad():
            for s_idx, sample in enumerate(samples_to_vis):
                img_tensor = sample['image'].unsqueeze(0).to(self.device)
                instances = sample.get('instances', [])

                outputs = self.model(img_tensor)
                heatmaps = outputs['heatmaps'][0].cpu().numpy()
                class_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()
                pafs = outputs['pafs'][0].cpu().numpy()

                preds = decode_spatial_instances_single(heatmaps, class_probs, pafs=pafs)
                scale = 256.0 / 64.0

                img_np = (sample['image'].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                sx = panel_w / 256.0
                sy = panel_h / 256.0

                # Panel 1: Ground Truth
                p_gt = cv2.resize(img_bgr, (panel_w, panel_h))
                for g_i, gt in enumerate(instances):
                    kps = gt['keypoints']
                    for u, v in limbs:
                        if u < len(kps) and v < len(kps) and kps[u][2] > 0 and kps[v][2] > 0:
                            cv2.line(p_gt, (int(kps[u][0]*sx), int(kps[u][1]*sy)), (int(kps[v][0]*sx), int(kps[v][1]*sy)), (0, 255, 0), 2, cv2.LINE_AA)
                    for k_i in range(min(num_k, len(kps))):
                        gx, gy, gv = kps[k_i]
                        if gv > 0:
                            cv2.circle(p_gt, (int(gx*sx), int(gy*sy)), 4, kp_colors[k_i % len(kp_colors)], -1, cv2.LINE_AA)
                cv2.rectangle(p_gt, (0, 0), (panel_w, 24), (30, 30, 30), -1)
                cv2.putText(p_gt, f"Sample {s_idx+1}: Ground Truth", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

                # Panel 2: Model Prediction
                p_pred = cv2.resize(img_bgr, (panel_w, panel_h))
                for p_i, pred in enumerate(preds):
                    kps = pred['keypoints']
                    for u, v in limbs:
                        if u < len(kps) and v < len(kps) and kps[u][2] > 0.10 and kps[v][2] > 0.10:
                            cv2.line(p_pred, (int(kps[u][0]*scale*sx), int(kps[u][1]*scale*sy)), (int(kps[v][0]*scale*sx), int(kps[v][1]*scale*sy)), (0, 140, 255), 2, cv2.LINE_AA)
                    for k_i in range(min(num_k, len(kps))):
                        pk = kps[k_i]
                        if pk[2] > 0.10:
                            cv2.circle(p_pred, (int(pk[0]*scale*sx), int(pk[1]*scale*sy)), 4, kp_colors[k_i % len(kp_colors)], -1, cv2.LINE_AA)
                cv2.rectangle(p_pred, (0, 0), (panel_w, 24), (30, 30, 30), -1)
                cv2.putText(p_pred, f"Epoch {epoch} Pred ({len(preds)} feet)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

                combined_row = np.hstack([p_gt, p_pred])
                panels.append(combined_row)

        if panels:
            final_vis = np.vstack(panels)
            out_path = vis_dir / f"epoch_{epoch:03d}_vis.jpg"
            cv2.imwrite(str(out_path), final_vis)
            print(f"\\n  \U0001f5bc\ufe0f [Epoch {epoch}] Saved test visualization -> {out_path}")
            try:
                from IPython.display import display, Image as IPImage
                display(IPImage(filename=str(out_path), width=580))
            except Exception:
                pass"""

    if "def visualize_epoch" not in content:
        if old_end in content:
            content = content.replace(old_end, vis_method, 1)
            print("Successfully added visualize_epoch method to ARShoeM1Trainer.")
        else:
            print("Warning: old_end not found!")

    with open(TRAINER_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    print("trainer_m1.py patched successfully!")


if __name__ == '__main__':
    patch_trainer()

