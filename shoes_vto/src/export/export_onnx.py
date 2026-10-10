"""
Export ARShoe M1v2 PyTorch Checkpoint to ONNX and Quantized INT8 Format

Exports:
1. Standard FP32 ONNX model (opset 12) with validation against PyTorch outputs.
2. Dynamic INT8 quantized ONNX model for low-latency WebAssembly / browser CPU execution.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

# Ensure paths are resolved properly
CURRENT_DIR = Path(__file__).resolve().parent
SHOES_VTO_DIR = CURRENT_DIR.parent.parent
SRC_DIR = CURRENT_DIR.parent
if str(SHOES_VTO_DIR) not in sys.path:
    sys.path.insert(0, str(SHOES_VTO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.arshoe_m1v2 import ARShoeM1v2


class ARShoeM1v2ExportWrapper(torch.nn.Module):
    """
    Wrapper module to guarantee standard tuple / list output format in ONNX
    and strictly opset-12 compliant operations.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # Return heatmaps, pafs, and class probability maps
        return out['heatmaps'], out['pafs'], out['class']


def load_model(checkpoint_path, device='cpu'):
    """Load ARShoeM1v2 weights from checkpoint with prefix cleanup."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ARShoeM1v2()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and any('s4.' in k for k in ckpt.keys()):
        state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned_sd = {}
    for k, v in state_dict.items():
        clean_k = k.replace('_orig_mod.', '').replace('module.', '')
        cleaned_sd[clean_k] = v

    model.load_state_dict(cleaned_sd)
    model.to(device)
    model.eval()
    return model


def export_onnx(
    checkpoint_path,
    output_dir,
    image_size=256,
    opset_version=12,
    dynamic_batch=True,
    quantize=True
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("📦 Exporting ARShoe M1v2 to Production ONNX")
    print("=" * 80)
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"  Output Dir  : {output_dir}")
    print(f"  Input Shape : [1, 3, {image_size}, {image_size}]")
    print(f"  Opset Ver   : {opset_version}")
    
    # 1. Load model
    model = load_model(checkpoint_path, device='cpu')
    wrapper = ARShoeM1v2ExportWrapper(model)
    wrapper.eval()
    
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    
    fp32_onnx_path = output_dir / "arshoe_m1v2.onnx"
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'heatmaps': {0: 'batch_size'},
            'pafs': {0: 'batch_size'},
            'class_probs': {0: 'batch_size'}
        }

    input_names = ['image']
    output_names = ['heatmaps', 'pafs', 'class_probs']

    # 2. PyTorch to ONNX export
    print(f"\n🚀 Exporting FP32 ONNX -> {fp32_onnx_path.name}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(fp32_onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    fp32_size_mb = fp32_onnx_path.stat().st_size / (1024 * 1024)
    print(f"✅ FP32 ONNX exported successfully! Size: {fp32_size_mb:.2f} MB")

    # 3. Verify ONNX validity with onnx package
    import onnx
    onnx_model = onnx.load(str(fp32_onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX graph check passed!")

    # 4. Verify numerical parity against PyTorch via ONNXRuntime
    import onnxruntime as ort
    ort_session = ort.InferenceSession(str(fp32_onnx_path), providers=['CPUExecutionProvider'])
    
    test_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    with torch.no_grad():
        pt_hm, pt_paf, pt_cls = wrapper(torch.from_numpy(test_input))
        pt_hm = pt_hm.numpy()
        pt_paf = pt_paf.numpy()
        pt_cls = pt_cls.numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_hm, ort_paf, ort_cls = ort_outs[0], ort_outs[1], ort_outs[2]

    diff_hm = np.max(np.abs(pt_hm - ort_hm))
    diff_paf = np.max(np.abs(pt_paf - ort_paf))
    diff_cls = np.max(np.abs(pt_cls - ort_cls))

    print(f"📊 Parity Verification (Max Absolute Error):")
    print(f"  • Heatmaps  : {diff_hm:.2e} (< 1e-4)")
    print(f"  • PAFs      : {diff_paf:.2e} (< 1e-4)")
    print(f"  • Class Probs: {diff_cls:.2e} (< 1e-4)")
    assert max(diff_hm, diff_paf, diff_cls) < 1e-4, "Numerical difference exceeded tolerance!"

    # 5. INT8 Dynamic Quantization
    int8_onnx_path = output_dir / "arshoe_m1v2_int8.onnx"
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            print(f"\n⚡ Generating INT8 Quantized model -> {int8_onnx_path.name}...")
            quantize_dynamic(
                model_input=str(fp32_onnx_path),
                model_output=str(int8_onnx_path),
                weight_type=QuantType.QUInt8
            )
            int8_size_mb = int8_onnx_path.stat().st_size / (1024 * 1024)
            print(f"✅ INT8 ONNX generated! Size: {int8_size_mb:.2f} MB (Shrunk by {((fp32_size_mb - int8_size_mb) / fp32_size_mb)*100:.1f}%)")
        except Exception as e:
            print(f"⚠️ INT8 quantization skipped: {e}")

    print("\n🎉 Model export completed successfully!")
    return fp32_onnx_path, int8_onnx_path if quantize else None


def main():
    parser = argparse.ArgumentParser(description="Export ARShoe M1v2 to ONNX")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/m1v2_training/checkpoints/best.pth',
        help='Path to PyTorch checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/exported_models',
        help='Directory to save exported ONNX models'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Input resolution'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=12,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--no_quantize',
        action='store_true',
        help='Disable INT8 quantization'
    )
    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        image_size=args.image_size,
        opset_version=args.opset,
        quantize=not args.no_quantize
    )


if __name__ == '__main__':
    main()

