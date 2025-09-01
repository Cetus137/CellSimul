import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for 256x256 cell membrane images.
    
    This discriminator classifies patches of the input image as real or fake,
    rather than classifying the entire image. This approach has been shown
    to produce sharper, more realistic images in GANs.
    
    Architecture:
    - Input: 256x256x1 grayscale images
    - Uses strided convolutions to downsample
    - Each layer reduces spatial dimensions by 2x
    - Outputs a patch-based prediction map
    """
    
    def __init__(self, in_channels=1, features=64):
        """
        Initialize the discriminator.
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale)
            features (int): Base number of features in first conv layer
        """
        super(Discriminator, self).__init__()
        
        self.in_channels = in_channels
        self.features = features
        
        # Define the discriminator layers
        # Input: 256x256x1
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1)  # 128x128x64
        self.leaky1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1)  # 64x64x128
        self.norm2 = nn.BatchNorm2d(features * 2)
        self.leaky2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1)  # 32x32x256
        self.norm3 = nn.BatchNorm2d(features * 4)
        self.leaky3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1)  # 16x16x512
        self.norm4 = nn.BatchNorm2d(features * 8)
        self.leaky4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv5 = nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1)  # 8x8x512
        self.norm5 = nn.BatchNorm2d(features * 8)
        self.leaky5 = nn.LeakyReLU(0.2, inplace=True)
        
        # Final classification layer
        self.conv6 = nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)  # 7x7x1
        # Note: No sigmoid here - we'll use BCEWithLogitsLoss for numerical stability

    def forward(self, x):
        """
        Forward pass of the discriminator.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 256, 256)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, 7, 7)
                         representing patch-based real/fake predictions
        """
        # Layer 1: 256x256 -> 128x128
        out = self.conv1(x)
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


