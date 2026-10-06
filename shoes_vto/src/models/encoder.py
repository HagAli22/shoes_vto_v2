"""
Fast-SCNN Encoder Backbone for ARShoe 16-Keypoint Model

Lightweight encoder based on Fast-SCNN architecture with depthwise separable convolutions.
Target: ~1.0-1.3M parameters, output stride = 4

Input:  [B, 3, 256, 256] RGB image
Output: [B, 128, 64, 64] feature maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution Block
    = Depthwise Conv + Pointwise Conv
    
    Reduces parameters and computation compared to standard convolution
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution (groups = in_channels means each channel is convolved separately)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels,
            bias=bias
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Pointwise convolution (1x1 conv to change channel dimension)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class DSConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block with optional stride
    Used as main building block in Fast-SCNN encoder
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(DSConvBlock, self).__init__()
        
        padding = 1 if stride == 1 else 1  # Maintain spatial resolution or downsample
        
        self.dsconv = DepthwiseSeparableConv(
            in_channels, out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding
        )
    
    def forward(self, x):
        return self.dsconv(x)


class FastSCNNEncoder(nn.Module):
    """
    Fast-SCNN Encoder for foot keypoint detection
    
    Architecture:
    - Initial Conv: 3 → 32 channels, stride=2 (128×128)
    - DSConv Block 1: 32 → 64 channels, stride=2 (64×64)
    - DSConv Block 2: 64 → 96 channels, stride=2 (32×32) 
    - DSConv Block 3: 96 → 128 channels, stride=2 (16×16)
    - Feature Extractor: 128 → 128 channels, upsample to 64×64
    
    Target parameter count: ~1.0-1.3M
    """
    
    def __init__(self, in_channels=3, output_channels=128):
        super(FastSCNNEncoder, self).__init__()
        
        # Initial convolution (stride=2 to downsample)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # After initial conv: 256×256 → 128×128
        
        # DSConv Block 1 (stride=2)
        self.dsconv1 = DSConvBlock(32, 64, stride=2)
        # After dsconv1: 128×128 → 64×64
        
        # DSConv Block 2 (stride=2)
        self.dsconv2 = DSConvBlock(64, 96, stride=2)
        # After dsconv2: 64×64 → 32×32
        
        # DSConv Block 3 (stride=2)
        self.dsconv3 = DSConvBlock(96, 128, stride=2)
        # After dsconv3: 32×32 → 16×16
        
        # Feature extraction head (expand receptive field)
        self.feature_extractor = nn.Sequential(
            DSConvBlock(128, 128, stride=1),  # Maintain spatial resolution
            DSConvBlock(128, output_channels, stride=1)
        )
        # After feature_extractor: 16×16 (same spatial size)
        
        # Upsample to output stride=4 (64×64 for 256×256 input)
        # This will be done in forward pass using interpolate
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, 3, 256, 256] input RGB image
        
        Returns:
            features: [B, 128, 64, 64] output feature maps
        """
        # Initial conv: 256×256 → 128×128
        x = self.initial_conv(x)
        
        # DSConv blocks with downsampling
        x = self.dsconv1(x)  # 128×128 → 64×64
        x_64 = x  # Save for potential skip connection
        
        x = self.dsconv2(x)  # 64×64 → 32×32
        x = self.dsconv3(x)  # 32×32 → 16×16
        
        # Feature extraction
        x = self.feature_extractor(x)  # 16×16 (128 channels)
        
        # Upsample to 64×64 (output stride=4)
        # Use bilinear interpolation
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        
        return x
    
    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_breakdown(self):
        """Get parameter count breakdown by layer"""
        breakdown = {}
        
        breakdown['initial_conv'] = sum(p.numel() for p in self.initial_conv.parameters() if p.requires_grad)
        breakdown['dsconv1'] = sum(p.numel() for p in self.dsconv1.parameters() if p.requires_grad)
        breakdown['dsconv2'] = sum(p.numel() for p in self.dsconv2.parameters() if p.requires_grad)
        breakdown['dsconv3'] = sum(p.numel() for p in self.dsconv3.parameters() if p.requires_grad)
        breakdown['feature_extractor'] = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
        breakdown['total'] = self.count_parameters()
        
        return breakdown


class FastSCNNEncoderV2(nn.Module):
    """
    Alternative Fast-SCNN Encoder with slightly different architecture
    This version uses output stride=4 directly without final upsampling
    
    Architecture:
    - Initial Conv: 3 → 32 channels, stride=2 (128×128)
    - DSConv Block 1: 32 → 64 channels, stride=2 (64×64)
    - DSConv Block 2: 64 → 96 channels, stride=1 (64×64) - NO downsampling
    - DSConv Block 3: 96 → 128 channels, stride=1 (64×64) - NO downsampling
    
    This keeps spatial resolution at 64×64 throughout stages 2-3
    Potentially more efficient than upsampling
    """
    
    def __init__(self, in_channels=3, output_channels=128):
        super(FastSCNNEncoderV2, self).__init__()
        
        # Initial convolution (stride=2)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # After initial conv: 256×256 → 128×128
        
        # DSConv Block 1 (stride=2 to reach 64×64)
        self.dsconv1 = DSConvBlock(32, 64, stride=2)
        # After dsconv1: 128×128 → 64×64
        
        # DSConv Block 2 (stride=1, maintain 64×64)
        self.dsconv2 = DSConvBlock(64, 96, stride=1)
        # After dsconv2: 64×64 (same)
        
        # DSConv Block 3 (stride=1, maintain 64×64)
        self.dsconv3 = DSConvBlock(96, output_channels, stride=1)
        # After dsconv3: 64×64 (128 channels)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, 3, 256, 256] input RGB image
        
        Returns:
            features: [B, 128, 64, 64] output feature maps
        """
        x = self.initial_conv(x)  # 256×256 → 128×128
        x = self.dsconv1(x)       # 128×128 → 64×64
        x = self.dsconv2(x)       # 64×64 (same)
        x = self.dsconv3(x)       # 64×64 (128 channels)
        
        return x
    
    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_breakdown(self):
        """Get parameter count breakdown by layer"""
        breakdown = {}
        
        breakdown['initial_conv'] = sum(p.numel() for p in self.initial_conv.parameters() if p.requires_grad)
        breakdown['dsconv1'] = sum(p.numel() for p in self.dsconv1.parameters() if p.requires_grad)
        breakdown['dsconv2'] = sum(p.numel() for p in self.dsconv2.parameters() if p.requires_grad)
        breakdown['dsconv3'] = sum(p.numel() for p in self.dsconv3.parameters() if p.requires_grad)
        breakdown['total'] = self.count_parameters()
        
        return breakdown


def create_encoder(version='v2'):
    """
    Factory function to create encoder
    
    Args:
        version: 'v1' or 'v2'
            v1: Uses 4 downsampling stages + upsample
            v2: Uses 2 downsampling stages + 2 same-resolution stages (more efficient)
    
    Returns:
        encoder: FastSCNNEncoder or FastSCNNEncoderV2
    """
    if version == 'v1':
        return FastSCNNEncoder()
    elif version == 'v2':
        return FastSCNNEncoderV2()
    else:
        raise ValueError(f"Unknown encoder version: {version}")


if __name__ == "__main__":
    # Test encoder
    print("=" * 80)
    print("Testing Fast-SCNN Encoder Backbone")
    print("=" * 80)
    
    # Test V1
    print("\n[Test 1] FastSCNNEncoder V1 (with upsampling)")
    encoder_v1 = FastSCNNEncoder()
    
    # Forward pass
    x = torch.randn(1, 3, 256, 256)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        features = encoder_v1(x)
    
    print(f"Output shape: {features.shape}")
    print(f"Expected: torch.Size([1, 128, 64, 64])")
    
    # Parameter count
    breakdown = encoder_v1.get_parameter_breakdown()
    print(f"\nParameter breakdown:")
    for name, count in breakdown.items():
        print(f"  {name}: {count:,}")
    
    print(f"\nTotal parameters: {breakdown['total']:,}")
    print(f"Target: ≤1,300,000 parameters")
    print(f"Status: {'✅ PASS' if breakdown['total'] <= 1_300_000 else '❌ FAIL'}")
    
    # Test V2
    print("\n" + "=" * 80)
    print("[Test 2] FastSCNNEncoder V2 (no upsampling, more efficient)")
    encoder_v2 = FastSCNNEncoderV2()
    
    with torch.no_grad():
        features_v2 = encoder_v2(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features_v2.shape}")
    
    breakdown_v2 = encoder_v2.get_parameter_breakdown()
    print(f"\nParameter breakdown:")
    for name, count in breakdown_v2.items():
        print(f"  {name}: {count:,}")
    
    print(f"\nTotal parameters: {breakdown_v2['total']:,}")
    print(f"Target: ≤1,300,000 parameters")
    print(f"Status: {'✅ PASS' if breakdown_v2['total'] <= 1_300_000 else '❌ FAIL'}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("Comparison:")
    print(f"V1 parameters: {breakdown['total']:,}")
    print(f"V2 parameters: {breakdown_v2['total']:,}")
    print(f"Difference: {abs(breakdown['total'] - breakdown_v2['total']):,}")
    print(f"\nRecommendation: V2 is more efficient (no upsampling) with fewer parameters")
    print("=" * 80)
