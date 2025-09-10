"""
Mask-Only Generator for CellSynthesis-style fluorescent image generation

This generator follows the original CellSynthesis approach:
- Input: Only distance masks (no noise vector)
- Output: Fluorescent images
- Architecture: U-Net style for deterministic mask-to-image translation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskOnlyGenerator(nn.Module):
    """
    Mask-only Generator network following CellSynthesis approach
    
    Takes only distance masks as input to generate fluorescent images
    No noise vector - deterministic mask-to-image translation
    """
    
    def __init__(self, mask_channels=1, output_channels=1, features=64):
        """
        Initialize the Mask-Only Generator
        
        Args:
            mask_channels (int): Number of channels in the conditioning mask (default: 1)
            output_channels (int): Number of output channels for generated image (default: 1)
            features (int): Base number of features (default: 64)
        """
        super(MaskOnlyGenerator, self).__init__()
        
        self.mask_channels = mask_channels
        self.output_channels = output_channels
        
        # U-Net style encoder (downsampling)
        self.enc1 = self._make_encoder_block(mask_channels, features, normalize=False)      # 256->128
        self.enc2 = self._make_encoder_block(features, features * 2)                        # 128->64
        self.enc3 = self._make_encoder_block(features * 2, features * 4)                    # 64->32
        self.enc4 = self._make_encoder_block(features * 4, features * 8)                    # 32->16
        self.enc5 = self._make_encoder_block(features * 8, features * 8)                    # 16->8
        self.enc6 = self._make_encoder_block(features * 8, features * 8)                    # 8->4
        self.enc7 = self._make_encoder_block(features * 8, features * 8)                    # 4->2
        self.enc8 = self._make_encoder_block(features * 8, features * 8, normalize=False)   # 2->1
        
        # U-Net style decoder (upsampling with skip connections)
        self.dec1 = self._make_decoder_block(features * 8, features * 8, dropout=True)      # 1->2
        self.dec2 = self._make_decoder_block(features * 16, features * 8, dropout=True)     # 2->4
        self.dec3 = self._make_decoder_block(features * 16, features * 8, dropout=True)     # 4->8
        self.dec4 = self._make_decoder_block(features * 16, features * 8)                   # 8->16
        self.dec5 = self._make_decoder_block(features * 16, features * 4)                   # 16->32
        self.dec6 = self._make_decoder_block(features * 8, features * 2)                    # 32->64
        self.dec7 = self._make_decoder_block(features * 4, features)                        # 64->128
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def _make_encoder_block(self, in_channels, out_channels, normalize=True):
        """Create an encoder block"""
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _make_decoder_block(self, in_channels, out_channels, dropout=False):
        """Create a decoder block"""
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, mask):
        """
        Forward pass through the generator
        
        Args:
            mask (torch.Tensor): Distance mask tensor of shape (batch_size, 1, 256, 256)
            
        Returns:
            torch.Tensor: Generated fluorescent image of shape (batch_size, 1, 256, 256)
        """
        # Encoder path
        e1 = self.enc1(mask)        # 256 -> 128
        e2 = self.enc2(e1)          # 128 -> 64
        e3 = self.enc3(e2)          # 64 -> 32
        e4 = self.enc4(e3)          # 32 -> 16
        e5 = self.enc5(e4)          # 16 -> 8
        e6 = self.enc6(e5)          # 8 -> 4
        e7 = self.enc7(e6)          # 4 -> 2
        e8 = self.enc8(e7)          # 2 -> 1
        
        # Decoder path with skip connections
        d1 = self.dec1(e8)                          # 1 -> 2
        d2 = self.dec2(torch.cat([d1, e7], 1))      # 2 -> 4
        d3 = self.dec3(torch.cat([d2, e6], 1))      # 4 -> 8
        d4 = self.dec4(torch.cat([d3, e5], 1))      # 8 -> 16
        d5 = self.dec5(torch.cat([d4, e4], 1))      # 16 -> 32
        d6 = self.dec6(torch.cat([d5, e3], 1))      # 32 -> 64
        d7 = self.dec7(torch.cat([d6, e2], 1))      # 64 -> 128
        
        # Final output
        output = self.final(torch.cat([d7, e1], 1)) # 128 -> 256
        
        return output


class SimpleMaskOnlyGenerator(nn.Module):
    """
    Simplified Mask-Only Generator for faster training
    
    Simpler architecture with fewer layers but same U-Net principle
    """
    
    def __init__(self, mask_channels=1, output_channels=1, features=64):
        """
        Initialize the Simple Mask-Only Generator
        
        Args:
            mask_channels (int): Number of channels in the conditioning mask (default: 1)
            output_channels (int): Number of output channels for generated image (default: 1)
            features (int): Base number of features (default: 64)
        """
        super(SimpleMaskOnlyGenerator, self).__init__()
        
        self.mask_channels = mask_channels
        self.output_channels = output_channels
        
        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(mask_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 256 -> 128
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 128 -> 64
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 64 -> 32
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 32 -> 16
        
        # Decoder (upsampling with skip connections)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(inplace=True)
        )  # 16 -> 32
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True)
        )  # 32 -> 64
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )  # 64 -> 128
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )  # 128 -> 256
        
    def forward(self, mask):
        """
        Forward pass through the generator
        
        Args:
            mask (torch.Tensor): Distance mask tensor of shape (batch_size, 1, 256, 256)
            
        Returns:
            torch.Tensor: Generated fluorescent image of shape (batch_size, 1, 256, 256)
        """
        # Encoder path
        e1 = self.enc1(mask)        # 256 -> 128
        e2 = self.enc2(e1)          # 128 -> 64
        e3 = self.enc3(e2)          # 64 -> 32
        e4 = self.enc4(e3)          # 32 -> 16
        
        # Decoder path with skip connections
        d1 = self.dec1(e4)                          # 16 -> 32
        d2 = self.dec2(torch.cat([d1, e3], 1))      # 32 -> 64
        d3 = self.dec3(torch.cat([d2, e2], 1))      # 64 -> 128
        d4 = self.dec4(torch.cat([d3, e1], 1))      # 128 -> 256
        
        return d4
