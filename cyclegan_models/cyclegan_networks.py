"""
CycleGAN for Mask-to-Fluorescent Image Translation

This implementation provides bidirectional translation between:
- Distance masks (synthetic) ↔ Fluorescent images (real)

Key components:
- Generator M2F: Masks → Fluorescent images  
- Generator F2M: Fluorescent images → Masks
- Discriminator D_F: Real vs fake fluorescent images
- Discriminator D_M: Real vs fake masks
- Cycle consistency loss to prevent memorization

Based on: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for CycleGAN generators"""
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class CycleGANGenerator(nn.Module):
    """
    CycleGAN Generator with residual blocks
    Can be used for both M2F (mask-to-fluorescent) and F2M (fluorescent-to-mask)
    """
    def __init__(self, input_channels=1, output_channels=1, n_residual_blocks=6):
        super(CycleGANGenerator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class CycleGANDiscriminator(nn.Module):
    """
    CycleGAN Discriminator (PatchGAN)
    Can be used for both D_F (fluorescent) and D_M (mask) domains
    """
    def __init__(self, input_channels=1):
        super(CycleGANDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, img):
        return self.model(img)


class CycleGANLoss:
    """Loss functions for CycleGAN training"""
    
    def __init__(self, cycle_loss_weight=10.0, identity_loss_weight=0.5):
        self.cycle_loss_weight = cycle_loss_weight
        self.identity_loss_weight = identity_loss_weight
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def adversarial_loss(self, predictions, target_is_real):
        """Standard adversarial loss"""
        if target_is_real:
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)
        return self.mse_loss(predictions, target)
    
    def cycle_consistency_loss(self, real_img, reconstructed_img):
        """Cycle consistency loss"""
        return self.l1_loss(reconstructed_img, real_img)
    
    def identity_loss(self, real_img, same_img):
        """Identity loss (optional)"""
        return self.l1_loss(same_img, real_img)
    
    def generator_loss(self, fake_pred, real_A, real_B, fake_B, fake_A, 
                      reconstructed_A, reconstructed_B, same_A=None, same_B=None):
        """
        Complete generator loss for CycleGAN
        
        Args:
            fake_pred: Discriminator prediction on fake images
            real_A, real_B: Real images from both domains
            fake_A, fake_B: Generated fake images
            reconstructed_A, reconstructed_B: Cycle-reconstructed images
            same_A, same_B: Identity mapping results (optional)
        """
        # Adversarial loss
        adv_loss = self.adversarial_loss(fake_pred, target_is_real=True)
        
        # Cycle consistency loss
        cycle_loss = (self.cycle_consistency_loss(real_A, reconstructed_A) + 
                     self.cycle_consistency_loss(real_B, reconstructed_B))
        
        # Identity loss (optional)
        identity_loss = 0
        if same_A is not None and same_B is not None:
            identity_loss = (self.identity_loss(real_A, same_A) + 
                           self.identity_loss(real_B, same_B))
        
        # Total generator loss
        total_loss = (adv_loss + 
                     self.cycle_loss_weight * cycle_loss + 
                     self.identity_loss_weight * identity_loss)
        
        return {
            'total_loss': total_loss,
            'adv_loss': adv_loss.item(),
            'cycle_loss': cycle_loss.item(),
            'identity_loss': identity_loss.item() if isinstance(identity_loss, torch.Tensor) else identity_loss
        }
    
    def discriminator_loss(self, real_pred, fake_pred):
        """Standard discriminator loss"""
        real_loss = self.adversarial_loss(real_pred, target_is_real=True)
        fake_loss = self.adversarial_loss(fake_pred, target_is_real=False)
        return (real_loss + fake_loss) * 0.5


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)
    return net
