"""
Conditional Generator for fluorescent microscopy image generation

This generator creates fluorescent cell images conditioned on binary cell membrane masks.
The generator takes both noise and binary mask as input to generate realistic cell images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator network for creating fluorescent cell images from binary masks
    
    Takes binary mask (condition) and noise vector as input to generate fluorescent images
    Output size: 256 x 256 pixels (single channel fluorescent)
    """
    
    def __init__(self, latent_dim=100, mask_channels=1, output_channels=1):
        """
        Initialize the Conditional Generator
        
        Args:
            latent_dim (int): Dimension of the latent noise vector (default: 100)
            mask_channels (int): Number of channels in the conditioning mask (default: 1)
            output_channels (int): Number of output channels for generated image (default: 1)
        """
        super(ConditionalGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.mask_channels = mask_channels
        self.output_channels = output_channels
        
        # Mask encoder - processes the binary mask condition
        self.mask_encoder = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(mask_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Noise projector - projects noise to spatial features
        self.noise_projector = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * 512),
            nn.BatchNorm1d(8 * 8 * 512),
            nn.ReLU(True)
        )
        
        # Fusion layer - combines mask features and noise features
        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),  # 512 + 512 = 1024
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder - generates the fluorescent image
        # Input: 8x8x512 -> Output: 256x256x1
        
        # 8x8 -> 16x16
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 16x16 -> 32x32
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 32x32 -> 64x64
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 64x64 -> 128x128
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 128x128 -> 256x256
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final layer to generate fluorescent image
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()  # Output range [-1, 1]
    
    def forward(self, noise, mask):
        """
        Forward pass of the conditional generator
        
        Args:
            noise (torch.Tensor): Random noise vector of shape (batch_size, latent_dim)
            mask (torch.Tensor): Binary mask condition of shape (batch_size, 1, 256, 256)
            
        Returns:
            torch.Tensor: Generated fluorescent image of shape (batch_size, 1, 256, 256)
        """
        batch_size = noise.size(0)
        
        # Encode the mask condition
        mask_features = self.mask_encoder(mask)  # (batch_size, 512, 8, 8)
        
        # Project noise to spatial features
        noise_projected = self.noise_projector(noise)  # (batch_size, 8*8*512)
        noise_features = noise_projected.view(batch_size, 512, 8, 8)  # (batch_size, 512, 8, 8)
        
        # Fuse mask and noise features
        combined_features = torch.cat([mask_features, noise_features], dim=1)  # (batch_size, 1024, 8, 8)
        fused_features = self.fusion(combined_features)  # (batch_size, 512, 8, 8)
        
        # Decode to generate fluorescent image
        x = self.decoder1(fused_features)  # 16x16
        x = self.decoder2(x)               # 32x32
        x = self.decoder3(x)               # 64x64
        x = self.decoder4(x)               # 128x128
        x = self.decoder5(x)               # 256x256
        
        # Final convolution and activation
        output = self.final_conv(x)
        output = self.tanh(output)
        
        return output


class SimpleConditionalGenerator(nn.Module):
    """
    Simplified conditional generator for faster experimentation
    """
    
    def __init__(self, latent_dim=100, mask_channels=1, output_channels=1):
        super(SimpleConditionalGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Simple approach: concatenate mask with noise-generated features
        self.noise_to_image = nn.Sequential(
            nn.Linear(latent_dim, 128 * 64 * 64),
            nn.BatchNorm1d(128 * 64 * 64),
            nn.ReLU(True)
        )
        
        # Process concatenated mask + noise features
        self.generator = nn.Sequential(
            # Input: mask (1 channel) + noise features (128 channels) = 129 channels at 64x64
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(129, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final layer
            nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise, mask):
        batch_size = noise.size(0)
        
        # Project noise to spatial features
        noise_features = self.noise_to_image(noise)
        noise_features = noise_features.view(batch_size, 128, 64, 64)
        
        # Downsample mask to match noise features
        mask_downsampled = F.interpolate(mask, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Concatenate mask and noise features
        combined = torch.cat([mask_downsampled, noise_features], dim=1)
        
        # Generate image
        output = self.generator(combined)
        
        return output


def test_conditional_generator():
    """
    Test function for the conditional generator
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test full conditional generator
    print("Testing ConditionalGenerator...")
    gen = ConditionalGenerator(latent_dim=100).to(device)
    
    # Create test inputs
    batch_size = 2
    noise = torch.randn(batch_size, 100, device=device)
    mask = torch.randint(0, 2, (batch_size, 1, 256, 256), dtype=torch.float32, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = gen(noise, mask)
    
    print(f"Input noise shape: {noise.shape}")
    print(f"Input mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test simple conditional generator
    print("\nTesting SimpleConditionalGenerator...")
    simple_gen = SimpleConditionalGenerator(latent_dim=100).to(device)
    
    with torch.no_grad():
        simple_output = simple_gen(noise, mask)
    
    print(f"Simple output shape: {simple_output.shape}")
    print(f"Simple output range: [{simple_output.min().item():.3f}, {simple_output.max().item():.3f}]")
    
    # Parameter count
    full_params = sum(p.numel() for p in gen.parameters())
    simple_params = sum(p.numel() for p in simple_gen.parameters())
    
    print(f"\nParameter counts:")
    print(f"ConditionalGenerator: {full_params:,}")
    print(f"SimpleConditionalGenerator: {simple_params:,}")
    
    print("\nConditional generators working correctly!")


if __name__ == "__main__":
    test_conditional_generator()
