'''
This file is the definition of the generator model used as part of the GAN
Generates synthetic cell membrane images of size 128x128
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generator network for creating synthetic cell membrane images
    Output size: 128 x 128 pixels
    """
    
    def __init__(self, latent_dim=100):
        """
        Initialize the Generator
        
        Args:
            latent_dim (int): Dimension of the latent noise vector (default: 100)
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # For 128x128 output, we can use clean power-of-2 upsampling
        # Start with 4x4 and upsample 5 times: 4→8→16→32→64→128
        
        self.init_size = 4  # Starting spatial size
        self.feature_maps = 512
        
        # Linear layer to project latent vector to initial feature map
        self.linear = nn.Linear(
            latent_dim, 
            self.feature_maps * self.init_size * self.init_size
        )
        
        # Convolutional transpose layers for upsampling
        self.conv_blocks = nn.Sequential(
            # Initial: 4x4x512
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Output: 8x8x256
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output: 16x16x128
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: 32x32x64
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output: 64x64x32
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Output: 128x128x16
        )
        
        # Final layer to get single channel output
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output values between -1 and 1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        """
        Forward pass of the generator
        
        Args:
            z (torch.Tensor): Latent noise vector of shape (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Generated images of shape (batch_size, 1, 128, 128)
        """
        batch_size = z.size(0)
        
        # Project and reshape
        x = self.linear(z)
        x = x.view(batch_size, self.feature_maps, self.init_size, self.init_size)
        
        # Apply convolutional blocks
        x = self.conv_blocks(x)
        # x is now (batch_size, 16, 128, 128)
        
        # Final convolution
        x = self.final_conv(x)
        # x is now (batch_size, 1, 128, 128)
        
        return x


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator that can take additional conditioning information
    such as cell density, pattern type, etc.
    """
    
    def __init__(self, latent_dim=100, num_classes=3, embedding_dim=50):
        """
        Initialize the Conditional Generator
        
        Args:
            latent_dim (int): Dimension of the latent noise vector
            num_classes (int): Number of conditioning classes (e.g., pattern types)
            embedding_dim (int): Dimension of class embedding
        """
        super(ConditionalGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Class embedding layer
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Combined input dimension
        input_dim = latent_dim + embedding_dim
        
        self.init_size = 4  # Starting spatial size for 128x128 output
        self.feature_maps = 512
        
        # Linear layer to project combined vector to initial feature map
        self.linear = nn.Linear(
            input_dim, 
            self.feature_maps * self.init_size * self.init_size
        )
        
        # Same convolutional blocks as base generator
        self.conv_blocks = nn.Sequential(
            # Initial: 4x4x512
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Output: 8x8x256
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output: 16x16x128
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: 32x32x64
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output: 64x64x32
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Output: 128x128x16
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    def forward(self, z, labels):
        """
        Forward pass of the conditional generator
        
        Args:
            z (torch.Tensor): Latent noise vector of shape (batch_size, latent_dim)
            labels (torch.Tensor): Class labels of shape (batch_size,)
            
        Returns:
            torch.Tensor: Generated images of shape (batch_size, 1, 128, 128)
        """
        batch_size = z.size(0)
        
        # Get class embeddings
        class_embed = self.class_embedding(labels)
        
        # Concatenate noise and class embedding
        combined_input = torch.cat([z, class_embed], dim=1)
        
        # Project and reshape
        x = self.linear(combined_input)
        x = x.view(batch_size, self.feature_maps, self.init_size, self.init_size)
        
        # Apply convolutional blocks
        x = self.conv_blocks(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x


def create_generator(generator_type='basic', **kwargs):
    """
    Factory function to create different types of generators
    
    Args:
        generator_type (str): Type of generator ('basic' or 'conditional')
        **kwargs: Additional arguments for generator initialization
        
    Returns:
        nn.Module: Generator instance
    """
    if generator_type == 'basic':
        return Generator(**kwargs)
    elif generator_type == 'conditional':
        return ConditionalGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


"""
=================================================================================
DETAILED CODE BREAKDOWN
=================================================================================

This file implements two generator architectures for creating synthetic cell 
membrane images of size 128x128 pixels with single channel (grayscale) output.

1. BASIC GENERATOR CLASS:
   ------------------------
   
   Input: Random noise vector of dimension 100 (latent_dim)
   Output: Single-channel image of size (batch_size, 1, 128, 128)
   
   Architecture Flow:
   
   Step 1: Linear Projection
   - Takes noise vector z of shape (batch_size, 100)
   - Projects to (batch_size, 512 * 4 * 4) = (batch_size, 8192)
   - Reshapes to (batch_size, 512, 4, 4)
   
   Step 2: Upsampling Sequence (5 ConvTranspose2d layers)
   - Layer 1: 512 → 256 channels, 4×4 → 8×8 (stride=2, kernel=4)
   - Layer 2: 256 → 128 channels, 8×8 → 16×16
   - Layer 3: 128 → 64 channels, 16×16 → 32×32
   - Layer 4: 64 → 32 channels, 32×32 → 64×64
   - Layer 5: 32 → 16 channels, 64×64 → 128×128
   
   Each upsampling layer includes:
   - ConvTranspose2d (upsamples by factor of 2)
   - BatchNorm2d (normalizes feature maps)
   - ReLU activation (introduces non-linearity)
   
   Step 3: Final Processing
   - Conv2d: 16 → 1 channel, maintains 128×128 size
   - Tanh activation: outputs values in range [-1, 1]
   
   Weight Initialization:
   - Convolutional layers: Normal distribution (mean=0, std=0.02)
   - BatchNorm layers: Normal weights (mean=1, std=0.02), zero bias
   - Linear layers: Normal distribution (mean=0, std=0.02)

2. CONDITIONAL GENERATOR CLASS:
   -----------------------------
   
   Similar architecture but with conditioning capability:
   
   Additional Features:
   - Class embedding layer for conditioning labels
   - Concatenates embedded labels with noise vector
   - Allows control over generation (e.g., tissue pattern type)
   
   Input: 
   - Noise vector z of shape (batch_size, 100)
   - Class labels of shape (batch_size,)
   
   Process:
   - Embeds class labels to 50-dimensional vectors
   - Concatenates with noise: (batch_size, 150)
   - Follows same upsampling architecture as basic generator
   
   Use Cases:
   - Generate specific tissue patterns (random, hexagonal, clustered)
   - Control cell density or other morphological features
   - Create diverse synthetic datasets with known labels

3. KEY DESIGN DECISIONS:
   ---------------------
   
   Target Size (128×128):
   - Perfect power of 2 (128 = 2^7)
   - Allows clean upsampling: 4→8→16→32→64→128
   - No cropping needed, simplifies architecture
   
   Single Channel Output:
   - Simplified from multi-channel to single grayscale channel
   - Suitable for cell membrane mask generation
   - Reduces model complexity and training time
   
   Activation Functions:
   - ReLU in hidden layers: prevents vanishing gradients
   - Tanh output: bounded output range [-1, 1]
   - Compatible with common GAN training practices
   
   Feature Map Progression:
   - Starts with high channels (512), reduces to 1
   - Allows rich feature learning in early layers
   - Gradual reduction prevents information bottlenecks

4. USAGE EXAMPLES:
   ---------------
   
   Basic Generation:
   ```python
   generator = Generator(latent_dim=100)
   noise = torch.randn(4, 100)
   fake_images = generator(noise)  # Shape: (4, 1, 128, 128)
   ```
   
   Conditional Generation:
   ```python
   cond_gen = ConditionalGenerator(latent_dim=100, num_classes=3)
   noise = torch.randn(4, 100)
   labels = torch.randint(0, 3, (4,))  # 0=random, 1=hexagonal, 2=clustered
   fake_images = cond_gen(noise, labels)  # Shape: (4, 1, 128, 128)
   ```

5. INTEGRATION WITH TRAINING:
   --------------------------
   
   This generator is designed to work with:
   - Standard GAN loss functions (BCE, Wasserstein, etc.)
   - Discriminator networks of compatible input size
   - Data loaders providing 128×128 real cell images
   - Common GAN training loops and optimizers (Adam, etc.)
   
   The single-channel output matches typical cell membrane imaging
   data and synthetic masks generated by the SyntheticTissue2D class.

=================================================================================
"""