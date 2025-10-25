"""
Enhanced Image Forgery Detection Model with Pre-trained Backbone
Uses ResNet50 pre-trained on ImageNet for better feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class PretrainedForgeryCNN(nn.Module):
    """
    Enhanced CNN with pre-trained ResNet50 backbone for superior performance
    """
    
    def __init__(self, pretrained=True, freeze_backbone_epochs=5):
        super(PretrainedForgeryCNN, self).__init__()
        
        # Load pre-trained ResNet50 with progress bar
        if pretrained:
            print("📦 Loading ResNet50 pre-trained weights...")
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=True)
            print("✅ Loaded ResNet50 pre-trained on ImageNet (1K v2)")
            print("🎯 Pre-trained features will accelerate convergence")
        else:
            self.backbone = models.resnet50(weights=None)
            print("⚠️  Using ResNet50 without pre-training (slower convergence)")
        
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.current_epoch = 0
        
        # Remove the final classification layer
        self.backbone_features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Feature extraction layers with different scales
        self.feature_dims = [64, 256, 512, 1024, 2048]  # ResNet50 feature dimensions
        
        # Enhanced decoder with skip connections and attention
        self.decoder = EnhancedDecoder(self.feature_dims)
        
        # Noise residual learning branch
        self.noise_branch = NoiseResidualBranch()
        
        # Final prediction layers (no sigmoid - BCEWithLogitsLoss handles it)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # Initialize decoder weights
        self._initialize_decoder()
    
    def _initialize_decoder(self):
        """Initialize decoder weights with Xavier initialization"""
        for module in [self.decoder, self.noise_branch, self.final_conv]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def set_epoch(self, epoch):
        """Set current epoch for backbone freezing control"""
        self.current_epoch = epoch
        
        # Freeze/unfreeze backbone based on epoch
        if epoch < self.freeze_backbone_epochs:
            for param in self.backbone_features.parameters():
                param.requires_grad = False
            print(f"Epoch {epoch}: Backbone FROZEN (fine-tuning decoder only)")
        else:
            for param in self.backbone_features.parameters():
                param.requires_grad = True
            print(f"Epoch {epoch}: Backbone UNFROZEN (end-to-end training)")
    
    def extract_multi_scale_features(self, x):
        """Extract features at multiple scales from ResNet backbone"""
        features = []
        
        # Initial convolution
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        features.append(x)  # 64 channels, 128x128
        
        x = self.backbone.maxpool(x)
        
        # ResNet blocks
        x = self.backbone.layer1(x)
        features.append(x)  # 256 channels, 64x64
        
        x = self.backbone.layer2(x)
        features.append(x)  # 512 channels, 32x32
        
        x = self.backbone.layer3(x)
        features.append(x)  # 1024 channels, 16x16
        
        x = self.backbone.layer4(x)
        features.append(x)  # 2048 channels, 8x8
        
        return features
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract multi-scale features
        features = self.extract_multi_scale_features(x)
        
        # Decode with skip connections
        decoded = self.decoder(features)
        
        # Noise residual learning
        noise_residual = self.noise_branch(x)
        
        # Ensure noise residual matches decoder spatial dimensions
        if noise_residual.size(-2) != decoded.size(-2) or noise_residual.size(-1) != decoded.size(-1):
            noise_residual = F.interpolate(noise_residual, size=(decoded.size(-2), decoded.size(-1)), 
                                         mode='bilinear', align_corners=False)
        
        # Combine decoded features with noise residual
        combined = decoded + noise_residual
        
        # Final prediction
        output = self.final_conv(combined)
        
        # Ensure output is same size as input
        if output.size(-2) != x.size(-2) or output.size(-1) != x.size(-1):
            output = F.interpolate(output, size=(x.size(-2), x.size(-1)), 
                                 mode='bilinear', align_corners=False)
        
        return output


class EnhancedDecoder(nn.Module):
    """Enhanced decoder with attention and skip connections"""
    
    def __init__(self, feature_dims):
        super(EnhancedDecoder, self).__init__()
        
        # Decoder blocks with attention
        self.decoder_blocks = nn.ModuleList()
        
        # From deepest to shallowest
        # feature_dims = [64, 256, 512, 1024, 2048]  # ResNet50 feature dimensions
        in_channels = [feature_dims[4], 512, 256, 128]  # 2048 -> 512 -> 256 -> 128
        skip_channels = [feature_dims[3], feature_dims[2], feature_dims[1], feature_dims[0]]  # 1024, 512, 256, 64
        out_channels = [512, 256, 128, 64]
        
        for i, (in_ch, skip_ch, out_ch) in enumerate(zip(in_channels, skip_channels, out_channels)):
            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch, use_attention=(i < 2))
            )
    
    def forward(self, features):
        # Start from deepest features
        x = features[-1]  # 2048 channels
        
        # Progressive upsampling with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_feat = features[-(i+2)]  # Corresponding skip connection
            x = decoder_block(x, skip_feat)
        
        return x


class DecoderBlock(nn.Module):
    """Decoder block with skip connection and optional attention"""
    
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=False):
        super(DecoderBlock, self).__init__()
        
        self.use_attention = use_attention
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 
                                         kernel_size=2, stride=2)
        
        # Skip connection processing
        self.skip_conv = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionGate(out_channels, out_channels, out_channels // 2)
        
        # Combined processing
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        
        # Process skip connection
        skip = self.skip_conv(skip)
        
        # Ensure spatial dimensions match
        if x.size(-2) != skip.size(-2) or x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        
        # Apply attention if enabled
        if self.use_attention:
            skip = self.attention(skip, x)
        
        # Concatenate and process
        combined = torch.cat([x, skip], dim=1)
        output = self.conv_block(combined)
        
        return output


class AttentionGate(nn.Module):
    """Attention gate mechanism for better feature selection"""
    
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Ensure spatial dimensions match
        if g1.size(-2) != x1.size(-2) or g1.size(-1) != x1.size(-1):
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class NoiseResidualBranch(nn.Module):
    """Specialized branch for learning noise residuals"""
    
    def __init__(self):
        super(NoiseResidualBranch, self).__init__()
        
        # High-pass filters for noise detection
        self.noise_filters = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.noise_filters(x)


def get_enhanced_model(pretrained=True, freeze_backbone_epochs=5):
    """
    Get enhanced forgery detection model with pre-trained backbone
    
    Args:
        pretrained: Whether to use pre-trained ResNet50
        freeze_backbone_epochs: Number of epochs to freeze backbone for fine-tuning
    
    Returns:
        Enhanced model instance
    """
    model = PretrainedForgeryCNN(pretrained=pretrained, 
                               freeze_backbone_epochs=freeze_backbone_epochs)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Enhanced Model Statistics:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model


if __name__ == "__main__":
    # Test model
    model = get_enhanced_model()
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print("✅ Model test passed!")