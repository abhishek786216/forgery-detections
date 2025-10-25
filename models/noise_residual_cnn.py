"""
Noise Residual CNN Model for Image Forgery Detection and Localization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseResidualBlock(nn.Module):
    """
    Noise Residual Block to extract noise patterns from images
    """
    def __init__(self, in_channels, out_channels):
        super(NoiseResidualBlock, self).__init__()
        
        # High-pass filter to extract noise residuals
        self.high_pass = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                  padding=1, bias=False)
        
        # Initialize high-pass filter with predefined weights
        self._init_high_pass_filter()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
        
    def _init_high_pass_filter(self):
        """Initialize high-pass filter for noise extraction"""
        # Simple high-pass filter kernel
        kernel = torch.tensor([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]], dtype=torch.float32)
        
        # Expand kernel to match input channels
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(self.high_pass.in_channels, 1, 1, 1)
        
        # Set the kernel weights
        with torch.no_grad():
            self.high_pass.weight.copy_(kernel)
        
        # Freeze the high-pass filter weights
        self.high_pass.weight.requires_grad = False
    
    def forward(self, x):
        # Extract noise residuals
        noise = self.high_pass(x)
        
        # Process through convolutional layers
        out = self.relu(self.bn1(self.conv1(noise)))
        out = self.bn2(self.conv2(out))
        
        # Skip connection
        skip = self.skip(x)
        
        # Add skip connection and apply ReLU
        out = self.relu(out + skip)
        
        return out


class ForgeryLocalizationCNN(nn.Module):
    """
    CNN Model for Image Forgery Detection and Localization using Noise Residual Learning
    """
    def __init__(self, input_channels=3, num_classes=1, use_sigmoid=True):
        super(ForgeryLocalizationCNN, self).__init__()
        self.use_sigmoid = use_sigmoid
        
        # Initial noise residual block
        self.initial_noise = NoiseResidualBlock(input_channels, 64)
        
        # Encoder (Downsampling path)
        self.encoder1 = self._make_encoder_block(64, 64)
        self.encoder2 = self._make_encoder_block(64, 128)
        self.encoder3 = self._make_encoder_block(128, 256)
        self.encoder4 = self._make_encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(512, 1024)
        
        # Decoder (Upsampling path)
        self.decoder4 = self._make_decoder_block(1024 + 512, 512)  # +512 for skip connection
        self.decoder3 = self._make_decoder_block(512 + 256, 256)   # +256 for skip connection
        self.decoder2 = self._make_decoder_block(256 + 128, 128)   # +128 for skip connection
        self.decoder1 = self._make_decoder_block(128 + 64, 64)     # +64 for skip connection
        
        # Final output layer
        self.final_conv = nn.Conv2d(64 + 64, num_classes, kernel_size=1)  # +64 for skip connection
        self.sigmoid = nn.Sigmoid()
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2, 2)
        
    def _make_encoder_block(self, in_channels, out_channels):
        """Create an encoder block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Initial noise residual extraction
        x0 = self.initial_noise(x)
        
        # Encoder path
        x1 = self.encoder1(x0)
        x1_pool = self.pool(x1)
        
        x2 = self.encoder2(x1_pool)
        x2_pool = self.pool(x2)
        
        x3 = self.encoder3(x2_pool)
        x3_pool = self.pool(x3)
        
        x4 = self.encoder4(x3_pool)
        x4_pool = self.pool(x4)
        
        # Bottleneck
        bottleneck = self.bottleneck(x4_pool)
        
        # Decoder path with skip connections
        d4 = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.decoder1(d1)
        
        # Final output with skip connection from initial layer
        final = torch.cat([d1, x0], dim=1)
        output = self.final_conv(final)
        
        # Apply sigmoid only if specified (for inference)
        if self.use_sigmoid:
            output = self.sigmoid(output)
        
        return output


def get_model(input_channels=3, num_classes=1, pretrained=False, use_sigmoid=True):
    """
    Get the Forgery Localization CNN model
    
    Args:
        input_channels (int): Number of input channels (default: 3 for RGB)
        num_classes (int): Number of output classes (default: 1 for binary segmentation)
        pretrained (bool): Whether to load pretrained weights (not implemented)
        use_sigmoid (bool): Whether to apply sigmoid at output (default: True for inference, False for training with BCE logits)
    
    Returns:
        model: ForgeryLocalizationCNN model
    """
    model = ForgeryLocalizationCNN(input_channels=input_channels, num_classes=num_classes, use_sigmoid=use_sigmoid)
    
    if pretrained:
        # TODO: Implement pretrained weights loading
        print("Pretrained weights not available yet.")
    
    return model


# Model summary function
def model_summary(model, input_size=(3, 256, 256)):
    """
    Print model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)
    
    print(f"Model Summary:")
    print(f"Input Size: {input_size}")
    
    try:
        output = model(dummy_input)
        print(f"Output Size: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model().to(device)
    
    # Print model summary
    model_summary(model)
    
    # Test forward pass
    test_input = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(test_input)
        print(f"Test output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")