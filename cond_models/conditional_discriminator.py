"""
Conditional Discriminator for fluorescent microscopy image evaluation

This discriminator evaluates both the generated fluorescent image and the conditioning mask
to determine if the fluorescent image is real or fake given the mask condition.
"""

import torch
import torch.nn as nn


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator for evaluating fluorescent images given binary masks
    
    Takes both fluorescent image and binary mask as input to classify real/fake
    Input: 256x256 fluorescent image + 256x256 binary mask
    """
    
    def __init__(self, image_channels=1, mask_channels=1, features=64):
        """
        Initialize the conditional discriminator
        
        Args:
            image_channels (int): Number of channels in fluorescent image (default: 1)
            mask_channels (int): Number of channels in conditioning mask (default: 1)
            features (int): Base number of features in first conv layer (default: 64)
        """
        super(ConditionalDiscriminator, self).__init__()
        
        self.image_channels = image_channels
        self.mask_channels = mask_channels
        self.features = features
        
        # Total input channels = image + mask
        total_input_channels = image_channels + mask_channels
        
        # Convolutional layers for processing combined input
        # Input: 256x256x2 (fluorescent + mask)
        
        # Layer 1: 256x256 -> 128x128
        self.conv1 = nn.Conv2d(total_input_channels, features, kernel_size=4, stride=2, padding=1)
        self.leaky1 = nn.LeakyReLU(0.2, inplace=True)
        
        # Layer 2: 128x128 -> 64x64
        self.conv2 = nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(features * 2)
        self.leaky2 = nn.LeakyReLU(0.2, inplace=True)
        
        # Layer 3: 64x64 -> 32x32
        self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.BatchNorm2d(features * 4)
        self.leaky3 = nn.LeakyReLU(0.2, inplace=True)
        
        # Layer 4: 32x32 -> 16x16
        self.conv4 = nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.BatchNorm2d(features * 8)
        self.leaky4 = nn.LeakyReLU(0.2, inplace=True)
        
        # Layer 5: 16x16 -> 8x8
        self.conv5 = nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1)
        self.norm5 = nn.BatchNorm2d(features * 8)
        self.leaky5 = nn.LeakyReLU(0.2, inplace=True)
        
        # Final classification layer: 8x8 -> 7x7
        self.conv6 = nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        # Note: No sigmoid here - we'll use BCEWithLogitsLoss for numerical stability

    def forward(self, image, mask):
        """
        Forward pass of the conditional discriminator
        
        Args:
            image (torch.Tensor): Fluorescent image tensor of shape (batch_size, 1, 256, 256)
            mask (torch.Tensor): Binary mask tensor of shape (batch_size, 1, 256, 256)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, 7, 7)
                         representing patch-based real/fake predictions
        """
        # Concatenate image and mask along channel dimension
        combined_input = torch.cat([image, mask], dim=1)  # (batch_size, 2, 256, 256)
        
        # Layer 1: 256x256 -> 128x128
        out = self.conv1(combined_input)
        out = self.leaky1(out)
        
        # Layer 2: 128x128 -> 64x64
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.leaky2(out)
        
        # Layer 3: 64x64 -> 32x32
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.leaky3(out)
        
        # Layer 4: 32x32 -> 16x16
        out = self.conv4(out)
        out = self.norm4(out)
        out = self.leaky4(out)
        
        # Layer 5: 16x16 -> 8x8
        out = self.conv5(out)
        out = self.norm5(out)
        out = self.leaky5(out)
        
        # Final layer: 8x8 -> 7x7
        out = self.conv6(out)
        
        return out


class SimpleConditionalDiscriminator(nn.Module):
    """
    Simplified conditional discriminator that outputs a single value per image
    """
    
    def __init__(self, image_channels=1, mask_channels=1, features=64):
        super(SimpleConditionalDiscriminator, self).__init__()
        
        total_input_channels = image_channels + mask_channels
        
        # Convolutional backbone
        self.backbone = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(total_input_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Global average pooling and final classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 8x8 -> 1x1
            nn.Conv2d(features * 8, 1, kernel_size=1)  # 1x1x1
        )
    
    def forward(self, image, mask):
        # Concatenate image and mask
        combined_input = torch.cat([image, mask], dim=1)
        
        # Extract features
        features = self.backbone(combined_input)
        
        # Classify
        output = self.classifier(features)
        output = output.view(output.size(0), -1)  # Flatten
        
        return output


def test_conditional_discriminator():
    """
    Test function for the conditional discriminator
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test patch-based conditional discriminator
    print("Testing ConditionalDiscriminator (patch-based)...")
    disc = ConditionalDiscriminator(image_channels=1, mask_channels=1, features=64).to(device)
    
    # Create test inputs
    batch_size = 2
    fluorescent_image = torch.randn(batch_size, 1, 256, 256, device=device)
    binary_mask = torch.randint(0, 2, (batch_size, 1, 256, 256), dtype=torch.float32, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = disc(fluorescent_image, binary_mask)
    
    print(f"Input fluorescent image shape: {fluorescent_image.shape}")
    print(f"Input binary mask shape: {binary_mask.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test simple conditional discriminator
    print("\nTesting SimpleConditionalDiscriminator...")
    simple_disc = SimpleConditionalDiscriminator(image_channels=1, mask_channels=1, features=64).to(device)
    
    with torch.no_grad():
        simple_output = simple_disc(fluorescent_image, binary_mask)
    
    print(f"Simple output shape: {simple_output.shape}")
    print(f"Simple output range: [{simple_output.min().item():.3f}, {simple_output.max().item():.3f}]")
    
    # Parameter count
    patch_params = sum(p.numel() for p in disc.parameters())
    simple_params = sum(p.numel() for p in simple_disc.parameters())
    
    print(f"\nParameter counts:")
    print(f"ConditionalDiscriminator (patch): {patch_params:,}")
    print(f"SimpleConditionalDiscriminator: {simple_params:,}")
    
    print("\nConditional discriminators working correctly!")


if __name__ == "__main__":
    test_conditional_discriminator()
