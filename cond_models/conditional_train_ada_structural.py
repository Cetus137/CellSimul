"""
Conditional GAN Training Script for Fluorescent Microscopy Images
Following Original CellSynthesis Approach: https://github.com/stegmaierj/CellSynthesis

This script trains a conditional GAN to generate fluorescent cell images
conditioned on binary cell membrane segmentation masks using PyTorch Lightning
with Adaptive Discriminator Augmentation (ADA) for stable training.

Uses simple CellSynthesis loss: adversarial_loss + identity_loss (no structural loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
import argparse
import json
from datetime import datetime
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

# Import our conditional models
from conditional_generator import ConditionalGenerator, SimpleConditionalGenerator
from conditional_discriminator import ConditionalDiscriminator, SimpleConditionalDiscriminator
from conditional_dataloader import ConditionalImageDataset, SingleDirectoryConditionalDataset
from unpaired_conditional_dataloader import UnpairedConditionalImageDataset


class MaskOnlyDataset(torch.utils.data.Dataset):
    """
    Dataset that loads only mask images for conditional generation
    """
    def __init__(self, mask_dir, image_size=256):
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        # Find all TIF files in mask directory
        import glob
        self.mask_paths = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            self.mask_paths.extend(glob.glob(os.path.join(mask_dir, ext)))
        
        if len(self.mask_paths) == 0:
            raise ValueError(f"No TIF files found in {mask_dir}")
        
        # Basic transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        print(f"Found {len(self.mask_paths)} mask files")
    
    def __len__(self):
        return len(self.mask_paths)
    
    def __getitem__(self, idx):
        # Load mask image
        mask_path = self.mask_paths[idx]
        
        try:
            # Try loading with tifffile first
            mask = tifffile.imread(mask_path)
            
            # Convert to PIL Image for transforms
            if mask.dtype != np.uint8:
                # Normalize to 0-255 range
                mask = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)
            
            mask = Image.fromarray(mask).convert('L')  # Convert to grayscale
            
        except Exception as e:
            print(f"Error loading {mask_path}: {e}")
            # Return a dummy image
            mask = Image.new('L', (self.image_size, self.image_size), 0)
        
        # Apply transforms
        mask = self.transform(mask)
        
        return mask
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        return {
            'num_masks': len(self.mask_paths),
            'mask_directory': self.mask_dir,
            'target_size': (self.image_size, self.image_size),
            'sample_paths': self.mask_paths[:5]
        }


class AdaptiveDiscriminatorAugmentation:
    """
    Ultra-Conservative Adaptive Discriminator Augmentation (ADA) for stable GAN training
    Prevents discriminator collapse by limiting maximum augmentation
    """
    def __init__(self, ada_target=0.6, ada_update=0.01, ada_update_period=4):
        self.ada_target = ada_target
        self.initial_target = ada_target  # Store initial target
        self.ada_update = ada_update
        self.ada_update_period = ada_update_period
        self.ada_aug_p = 0.0
        self.ada_step = 0
        self.ada_stats = []
        self.epoch_count = 0
        self.max_aug_prob = 0.7  # Conservative cap at 70%
        
    def update(self, real_predictions, epoch=None):
        """Update ADA probability based on discriminator accuracy on real images"""
        if epoch is not None:
            self.epoch_count = epoch
            # Keep target stable to prevent runaway augmentation
            self.ada_target = self.initial_target  # No dynamic increase
            
        with torch.no_grad():
            # Calculate accuracy on real images
            real_accuracy = (torch.sigmoid(real_predictions) > 0.5).float().mean().item()
            self.ada_stats.append(real_accuracy)
            
            if len(self.ada_stats) >= self.ada_update_period:
                mean_accuracy = np.mean(self.ada_stats)
                
                # Adjust augmentation probability based on accuracy with conservative cap
                if mean_accuracy > self.ada_target:
                    self.ada_aug_p = min(self.max_aug_prob, self.ada_aug_p + self.ada_update)  # Cap at max_aug_prob
                else:
                    self.ada_aug_p = max(0.0, self.ada_aug_p - self.ada_update)
                
                self.ada_stats = []
                
        return self.ada_aug_p


class StructuralLoss(nn.Module):
    """Improved structural loss to enforce meaningful mask-fluorescent correspondence"""
    def __init__(self):
        super(StructuralLoss, self).__init__()
        self._debug_count = 0
        
    def forward(self, fluorescent, masks):
        """
        Revolutionary structural loss designed to vary dramatically based on actual conditioning quality:
        1. Raw correlation without normalization (preserves natural variation)
        2. Spatial gradient alignment (detects edge correspondence)  
        3. Intensity distribution matching (histogram-based)
        """
        batch_size = fluorescent.size(0)
        
        if self._debug_count <= 3:
            print(f"DEBUG Structural Loss:")
            print(f"  Mask range: [{masks.min().item():.4f}, {masks.max().item():.4f}]")
            print(f"  Fluorescent range: [{fluorescent.min().item():.4f}, {fluorescent.max().item():.4f}]")
            print(f"  Mask std: {masks.std().item():.4f}")
            print(f"  Fluorescent std: {fluorescent.std().item():.4f}")
            self._debug_count += 1
        
        # Approach 1: Raw Pearson correlation (no normalization to preserve variation)
        # This should vary dramatically: 0.0 for random images, 1.0+ for perfect correlation
        correlation_loss = 0.0
        for b in range(batch_size):
            mask_flat = masks[b].view(-1)
            fluor_flat = fluorescent[b].view(-1)
            
            # Remove mean for correlation calculation
            mask_centered = mask_flat - mask_flat.mean()
            fluor_centered = fluor_flat - fluor_flat.mean()
            
            # Pearson correlation coefficient
            numerator = (mask_centered * fluor_centered).sum()
            denominator = torch.sqrt((mask_centered ** 2).sum() * (fluor_centered ** 2).sum())
            
            if denominator > 1e-8:
                correlation = numerator / denominator
                # Convert to loss: 0 for perfect correlation, higher for poor correlation
                correlation_loss += (1.0 - correlation) ** 2
            else:
                correlation_loss += 2.0  # Maximum penalty for degenerate case
        
        correlation_loss = correlation_loss / batch_size
        
        # Approach 2: Spatial gradient alignment
        # Good conditioning should have aligned gradients between mask and fluorescent
        def compute_gradients(x):
            grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])  # Horizontal gradients
            grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])  # Vertical gradients
            return grad_x, grad_y
        
        mask_grad_x, mask_grad_y = compute_gradients(masks)
        fluor_grad_x, fluor_grad_y = compute_gradients(fluorescent)
        
        # Cosine similarity between gradient vectors
        grad_alignment_x = F.cosine_similarity(mask_grad_x.view(batch_size, -1), 
                                              fluor_grad_x.view(batch_size, -1), dim=1)
        grad_alignment_y = F.cosine_similarity(mask_grad_y.view(batch_size, -1), 
                                              fluor_grad_y.view(batch_size, -1), dim=1)
        
        gradient_loss = (2.0 - grad_alignment_x.mean() - grad_alignment_y.mean())
        
        # Approach 3: Intensity distribution matching using Earth Mover's Distance approximation
        # Create histograms and compare distributions
        histogram_loss = 0.0
        for b in range(batch_size):
            # Create normalized histograms
            mask_hist = torch.histc(masks[b], bins=50, min=-1.0, max=1.0)
            fluor_hist = torch.histc(fluorescent[b], bins=50, min=-1.0, max=1.0)
            
            # Normalize to probability distributions
            mask_hist = mask_hist / (mask_hist.sum() + 1e-8)
            fluor_hist = fluor_hist / (fluor_hist.sum() + 1e-8)
            
            # Wasserstein-1 distance approximation (cumulative distribution difference)
            mask_cdf = torch.cumsum(mask_hist, dim=0)
            fluor_cdf = torch.cumsum(fluor_hist, dim=0)
            
            histogram_loss += torch.abs(mask_cdf - fluor_cdf).mean()
        
        histogram_loss = histogram_loss / batch_size
        
        # Combine with different weights to ensure variation
        # Correlation loss dominates (0-4 range), gradients add detail (0-2 range), histogram adds distribution matching (0-1 range)
        total_loss = 2.0 * correlation_loss + 1.0 * gradient_loss + 0.5 * histogram_loss
        
        if self._debug_count <= 3:
            print(f"  Correlation loss: {correlation_loss.item():.4f}")
            print(f"  Gradient loss: {gradient_loss.item():.4f}")
            print(f"  Histogram loss: {histogram_loss.item():.4f}")
            print(f"  Total loss: {total_loss.item():.4f}")
        
        return total_loss


class AdversarialLoss(nn.Module):
    """Simple adversarial loss using binary cross entropy"""
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        
    def forward(self, predictions, target_is_real):
        if target_is_real:
            targets = torch.ones_like(predictions)
        else:
            targets = torch.zeros_like(predictions)
        
        return F.binary_cross_entropy_with_logits(predictions, targets)


class IdentityLoss(nn.Module):
    """Identity loss using L1 distance"""
    def __init__(self):
        super(IdentityLoss, self).__init__()
        
    def forward(self, generated, target):
        return F.l1_loss(generated, target)


class ConditionalGANTrainer:
    """
    Conditional GAN Trainer class using CellSynthesis-inspired training approach
    with structural loss for mask conditioning
    """
    
    def __init__(self, fluorescent_dir, mask_dir, latent_dim=100, image_size=256, 
                 lr_g=0.0002, lr_d=0.0002, device=None, use_simple_models=False,
                 ada_target=0.1, ada_update=0.05):
        """
        Initialize the conditional GAN trainer with structural loss
        """
        self.fluorescent_dir = fluorescent_dir
        self.mask_dir = mask_dir
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.use_simple_models = use_simple_models
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize dataset for unpaired conditional generation
        # Masks as conditions, real fluorescent images for discriminator
        try:
            # Use UnpairedConditionalImageDataset with correct directories
            self.dataset = UnpairedConditionalImageDataset(
                fluorescent_dir=fluorescent_dir,  # Real fluorescent images
                mask_dir=mask_dir,                # Distance masks as conditions
                image_size=image_size,
                max_images=256  # Limit to 256 as requested
            )
            print("Using unpaired conditional dataset (masks as conditions, real fluorescent for discrimination)")
            
            # Validate data counts
            dataset_info = self.dataset.get_dataset_info()
            print(f"Dataset Info:")
            for key, value in dataset_info.items():
                print(f"  {key}: {value}")
            
            # Check if we have the expected 256 images
            if dataset_info['num_fluorescent'] != 256 or dataset_info['num_masks'] != 256:
                print(f"WARNING: Expected 256 images each, got {dataset_info['num_fluorescent']} fluorescent and {dataset_info['num_masks']} masks")
                
        except Exception as e:
            print(f"Error with unpaired dataset: {e}")
            print("Trying mask-only approach...")
            # Fallback to mask-only if unpaired doesn't work
            self.dataset = MaskOnlyDataset(mask_dir, image_size)
        
        # Initialize models
        if use_simple_models:
            self.generator = SimpleConditionalGenerator(latent_dim=latent_dim).to(self.device)
            self.discriminator = SimpleConditionalDiscriminator().to(self.device)
            print("Using simplified conditional models")
        else:
            self.generator = ConditionalGenerator(latent_dim=latent_dim).to(self.device)
            self.discriminator = ConditionalDiscriminator().to(self.device)
            print("Using full conditional models")
        
        # Initialize weights
        self._init_weights()
        
        # Loss functions - Following original CellSynthesis (adversarial + identity only)
        self.adversarial_loss = AdversarialLoss()
        self.identity_loss = IdentityLoss()
        # self.structural_loss = StructuralLoss()  # REMOVED: Not used in original CellSynthesis
        
        # Optimizers - CellSynthesis RAdam with increased learning rates for recovery
        # Use the passed learning rates or recovery-friendly defaults
        lr_g = self.lr_g if self.lr_g else 0.0002  # Increased from 0.0001
        lr_d = self.lr_d if self.lr_d else 0.0002  # Increased from 0.0001
        
        try:
            # Try to import RAdam from CellSynthesis approach
            from torch.optim import RAdam
            self.optimizer_G = RAdam(self.generator.parameters(), lr=lr_g, weight_decay=1e-4)
            self.optimizer_D = RAdam(self.discriminator.parameters(), lr=lr_d, weight_decay=1e-4)
            print(f"Using RAdam optimizer with ultra-conservative LR: G={lr_g}, D={lr_d}")
        except ImportError:
            # Fallback to Adam with CellSynthesis-like settings
            self.optimizer_G = torch.optim.Adam(
                self.generator.parameters(), 
                lr=lr_g, 
                betas=(0.5, 0.999),  # Standard GAN betas
                weight_decay=1e-4
            )
            self.optimizer_D = torch.optim.Adam(
                self.discriminator.parameters(), 
                lr=lr_d, 
                betas=(0.5, 0.999),
                weight_decay=1e-4
            )
            print(f"Using Adam optimizer with ultra-conservative LR: G={lr_g}, D={lr_d}")
            
        # Store the actual LRs used
        self.actual_lr_g = lr_g
        self.actual_lr_d = lr_d
        
        # Initialize ADA with ultra-conservative settings  
        self.ada_target = ada_target
        self.ada_update = ada_update
        
        print(f"Initialized ConditionalGANTrainer with CellSynthesis-inspired approach:")
        print(f"  Generator LR: {lr_g}")
        print(f"  Discriminator LR: {lr_d}")
        print(f"  ADA Target: {ada_target}")
        print(f"  ADA Update: {ada_update}")
        print(f"  ADA Update Period: 4 epochs (more stable)")
        print(f"  Device: {self.device}")
        
        self.ada = AdaptiveDiscriminatorAugmentation(
            ada_target=ada_target,
            ada_update=ada_update,
            ada_update_period=4  # Update every 4 epochs for stability (not every epoch)
        )
        
        # Training state
        self.current_epoch = 0
        self.g_losses = []
        self.d_losses = []
        self.ada_probs = []
    
    def _init_weights(self):
        """Initialize network weights"""
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        
        self.generator.apply(init_func)
        self.discriminator.apply(init_func)
        print("Initialized model weights")
    
    def training_step(self, batch, optimizer_idx):
        """
        Training step for unpaired conditional generation
        Generate fluorescent images from masks, discriminate against real fluorescent
        
        Args:
            batch: Batch containing (real_fluorescent, masks) from unpaired dataset
            optimizer_idx: 0 for generator, 1 for discriminator
        """
        real_fluorescent, masks = batch  # Unpaired real fluorescent and masks
        batch_size = masks.size(0)
        
        # Generate random noise
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        
        # Generate fake fluorescent images from masks
        fake_fluorescent = self.generator(noise, masks)
        
        if optimizer_idx == 0:
            # Generator training step
            return self._generator_step(fake_fluorescent, masks)
        else:
            # Discriminator training step
            return self._discriminator_step(fake_fluorescent.detach(), real_fluorescent, masks)
    
    def _compute_mask_dependency_loss(self, generated_images, masks):
        """
        Compute loss to ensure generated images depend on mask structure
        This prevents the generator from ignoring mask conditioning
        """
        # Compute spatial correlation between generated images and masks
        batch_size = generated_images.size(0)
        
        # Flatten spatial dimensions for correlation
        gen_flat = generated_images.view(batch_size, -1)  # [B, H*W]
        mask_flat = masks.view(batch_size, -1)  # [B, H*W]
        
        # Compute correlation coefficient for each sample
        correlations = []
        for i in range(batch_size):
            # Center the data
            gen_centered = gen_flat[i] - gen_flat[i].mean()
            mask_centered = mask_flat[i] - mask_flat[i].mean()
            
            # Compute correlation
            numerator = torch.sum(gen_centered * mask_centered)
            denominator = torch.sqrt(torch.sum(gen_centered**2) * torch.sum(mask_centered**2)) + 1e-8
            correlation = numerator / denominator
            correlations.append(correlation)
        
        # Average correlation across batch
        avg_correlation = torch.stack(correlations).mean()
        
        # Loss = 1 - |correlation| (we want high absolute correlation)
        mask_dependency_loss = 1.0 - torch.abs(avg_correlation)
        
        return mask_dependency_loss
    
    def _generator_step(self, fake_fluorescent, masks):
        """
        Generator training step following CellSynthesis methodology:
        - Adversarial loss (fool discriminator)
        - Identity loss (when feeding real images through generator)
        - Structural loss (mask-fluorescent correspondence)
        """
        # UNPAIRED CONDITIONAL GAN APPROACH
        # 1. Adversarial loss - fool the discriminator
        fake_pred = self.discriminator(fake_fluorescent, masks)
        g_adv_loss = self.adversarial_loss(fake_pred, target_is_real=True)
        
        # 2. Self-consistency loss - same inputs should produce same outputs
        # This ensures the generator is deterministic for given (noise, mask) pairs
        noise_consistency = torch.randn(fake_fluorescent.size(0), self.latent_dim, device=self.device)
        consistent_generated1 = self.generator(noise_consistency, masks)
        consistent_generated2 = self.generator(noise_consistency, masks)  # Same inputs
        g_consistency_loss = self.identity_loss(consistent_generated1, consistent_generated2)
        
        # Weighted combination: emphasize adversarial training for unpaired data
        g_loss = g_adv_loss + 0.1 * g_consistency_loss  # Light consistency regularization
        
        return {
            'loss': g_loss,
            'g_adv_loss': g_adv_loss.item(),
            'g_consistency_loss': g_consistency_loss.item()
        }
    
    def _discriminator_step(self, fake_fluorescent, real_fluorescent, masks):
        """
        Discriminator training step following CellSynthesis methodology:
        - Real images with proper conditioning
        - Fake images with proper conditioning
        - Balanced loss computation
        """
        
        # Real fluorescent images - for unpaired training, discriminate real images
        # Use the masks from the batch for conditioning (even though unpaired)
        # This creates a more balanced training dynamic
        batch_size = real_fluorescent.size(0)
        
        # For real images, we'll use a mix: some with zero masks, some with actual masks
        # This prevents the discriminator from simply learning to distinguish based on mask presence
        if torch.rand(1).item() > 0.5:
            # 50% chance: use zero masks for real images (unconditional discrimination)
            real_masks = torch.zeros_like(masks)
        else:
            # 50% chance: use actual masks for real images
            real_masks = masks
            
        # CellSynthesis-style discriminator training with label smoothing for stability
        # Real images: use label smoothing (0.9 instead of 1.0) to improve stability
        real_pred = self.discriminator(real_fluorescent, real_masks)
        real_labels = torch.ones_like(real_pred) * 0.9  # Label smoothing
        d_real_loss = F.binary_cross_entropy_with_logits(real_pred, real_labels)
        
        # Fake images: use hard labels (0.0) to maintain discrimination
        fake_pred = self.discriminator(fake_fluorescent, masks)
        fake_labels = torch.zeros_like(fake_pred)
        d_fake_loss = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)
        
        # Add mask-mismatch penalty to encourage conditioning awareness
        # If fake image has high intensity where mask is low, penalize discriminator
        # This forces discriminator to see mask-fluorescent correlations
        mask_mismatch_penalty = self._compute_mask_mismatch_penalty(fake_fluorescent, masks)
        
        # CellSynthesis-style discriminator loss: average of real and fake losses + mask penalty
        d_loss = (d_real_loss + d_fake_loss) / 2.0 + 0.1 * mask_mismatch_penalty
        
        # Update ADA based on real predictions
        ada_prob = self.ada.update(real_pred, epoch=self.current_epoch)
        
        # Emergency ADA reset if discriminator becomes too weak
        if d_loss.item() < 0.2:  # If discriminator loss is very low
            self.ada.ada_aug_p = max(0.0, self.ada.ada_aug_p - 0.1)  # Reduce augmentation quickly
            if d_loss.item() < 0.15:  # If extremely low
                self.ada.ada_aug_p = 0.0  # Reset to no augmentation
                print(f"WARNING: Discriminator too weak (loss={d_loss.item():.4f}), resetting ADA to 0.0")
        
        return {
            'loss': d_loss,
            'd_real_loss': d_real_loss.item(),
            'd_fake_loss': d_fake_loss.item(),
            'ada_prob': ada_prob
        }
    
    def _compute_mask_mismatch_penalty(self, fake_fluorescent, masks):
        """
        Compute penalty when fake fluorescent images don't match mask conditioning.
        This encourages the discriminator to notice when images violate mask structure.
        """
        # Normalize both to [0, 1] for comparison
        fake_norm = (fake_fluorescent - fake_fluorescent.min()) / (fake_fluorescent.max() - fake_fluorescent.min() + 1e-8)
        mask_norm = (masks - masks.min()) / (masks.max() - masks.min() + 1e-8)
        
        # Compute mismatch: high intensity where mask is low is bad
        # Use a threshold to identify mismatches
        fake_high = (fake_norm > 0.5).float()
        mask_low = (mask_norm < 0.3).float()
        
        # Penalty for high fluorescent signal where mask indicates it shouldn't be
        mismatch = (fake_high * mask_low).mean()
        
        return mismatch
    
    def _compute_gradient_penalty(self, real_images, fake_images, masks):
        """Compute gradient penalty for WGAN-GP style regularization"""
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        # Interpolate between real and fake images
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)
        
        # Get discriminator output for interpolated images
        # Use zero masks for interpolated samples to avoid mask conditioning issues
        d_interpolated = self.discriminator(interpolated, torch.zeros_like(masks))
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated.sum(),
            inputs=interpolated,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch using CellSynthesis-style alternating optimization
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_ada_prob = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device (real_fluorescent, masks from unpaired dataset)
            real_fluorescent, masks = [x.to(self.device) for x in batch]
            batch = (real_fluorescent, masks)
            
            # CellSynthesis-style training: alternating G and D updates
            # Train generator first (optimizer_idx=0)
            self.optimizer_G.zero_grad()
            g_results = self.training_step(batch, optimizer_idx=0)
            g_results['loss'].backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.optimizer_G.step()
            
            # Train discriminator second (optimizer_idx=1)  
            self.optimizer_D.zero_grad()
            d_results = self.training_step(batch, optimizer_idx=1)
            d_results['loss'].backward()
            
            # Gradient clipping for discriminator too
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.optimizer_D.step()
            
            # Accumulate losses
            epoch_g_loss += g_results['loss'].item()
            epoch_d_loss += d_results['loss'].item()
            epoch_ada_prob += d_results['ada_prob']
            
            # Print progress with unpaired conditional GAN metrics
            if batch_idx % 50 == 0:
                current_ada_target = self.ada.ada_target
                print(f"Batch {batch_idx}/{num_batches}: "
                      f"G_loss: {g_results['loss'].item():.4f} "
                      f"(adv: {g_results['g_adv_loss']:.4f}, "
                      f"cons: {g_results['g_consistency_loss']:.4f}), "
                      f"D_loss: {d_results['loss'].item():.4f} "
                      f"(real: {d_results['d_real_loss']:.4f}, "
                      f"fake: {d_results['d_fake_loss']:.4f}), "
                      f"ADA: {d_results['ada_prob']:.3f} "
                      f"(target: {current_ada_target:.2f})")
        
        # Average losses for the epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_ada_prob = epoch_ada_prob / num_batches
        
        # Store losses for tracking
        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)
        self.ada_probs.append(avg_ada_prob)
        
        return {
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'ada_prob': avg_ada_prob
        }
    
    def train(self, num_epochs, batch_size=16, save_interval=10, output_dir='outputs'):
        """
        Train the conditional GAN
        
        Args:
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size
            save_interval (int): How often to save samples and models
            output_dir (str): Directory to save outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create data loader
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {len(dataloader)}")
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            epoch_results = self.train_epoch(dataloader)
            
            # Print epoch results with dynamic ADA target info
            current_ada_target = self.ada.ada_target
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Generator Loss: {epoch_results['g_loss']:.6f}")
            print(f"  Discriminator Loss: {epoch_results['d_loss']:.6f}")
            print(f"  ADA Probability: {epoch_results['ada_prob']:.4f} (target: {current_ada_target:.2f})")
            
            # Save samples and models periodically
            if (epoch + 1) % save_interval == 0:
                self.save_samples(output_dir, epoch)
                self.display_samples(epoch)  # Display samples in addition to saving
                self.save_models(output_dir, epoch)
                self.save_training_curves(output_dir)
        
        print("Training completed!")
        self.save_final_results(output_dir)
    
    def save_samples(self, output_dir, epoch):
        """Save generated fluorescent samples from masks"""
        self.generator.eval()
        
        with torch.no_grad():
            # Get a batch of (real_fluorescent, masks) from dataset
            sample_batch = next(iter(DataLoader(self.dataset, batch_size=8, shuffle=True)))
            real_fluorescent, masks = [x.to(self.device) for x in sample_batch]
            
            # Data validation
            print(f"Sample batch shapes: real_fluorescent: {real_fluorescent.shape}, masks: {masks.shape}")
            print(f"Real fluorescent range: [{real_fluorescent.min():.3f}, {real_fluorescent.max():.3f}]")
            print(f"Masks range: [{masks.min():.3f}, {masks.max():.3f}]")
            
            # Generate fluorescent images from masks
            noise = torch.randn(masks.size(0), self.latent_dim, device=self.device)
            fake_fluorescent = self.generator(noise, masks)
            
            print(f"Generated fluorescent range: [{fake_fluorescent.min():.3f}, {fake_fluorescent.max():.3f}]")
            
            # Create grayscale overlay visualization 
            # Normalize both to [0,1] for proper overlay
            masks_norm = (masks + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
            fake_norm = (fake_fluorescent + 1.0) / 2.0
            
            # Create simple grayscale overlay by averaging normalized mask and generated
            # Bright areas = good alignment, dark areas = poor alignment
            overlay = (masks_norm + fake_norm) / 2.0
            
            # Save comparison: real_fluorescent -> masks -> generated -> grayscale overlay
            comparison = torch.cat([
                real_fluorescent.cpu(),
                masks.cpu(),
                fake_fluorescent.cpu(),
                overlay.cpu()
            ], dim=0)
            
            save_path = os.path.join(output_dir, f'samples_epoch_{epoch+1}.png')
            save_image(comparison, save_path, nrow=masks.size(0), normalize=True)
            print(f"Saved samples to {save_path}")
            print(f"  Row 1: Real fluorescent images (targets)")
            print(f"  Row 2: Distance masks (conditions - white = far from membrane)")
            print(f"  Row 3: Generated fluorescent images (from masks)")
            print(f"  Row 4: Grayscale Overlay (bright = good alignment, dark = poor alignment)")
    
    def display_samples(self, epoch):
        """Display generated samples in a matplotlib window"""
        # Ensure matplotlib displays inline in Colab
        try:
            import matplotlib
            matplotlib.use('inline')  # For Colab/Jupyter
        except:
            pass
            
        self.generator.eval()
        
        with torch.no_grad():
            # Get test samples
            test_dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=8, shuffle=True
            )
            batch = next(iter(test_dataloader))
            real_fluorescent, masks = batch
            real_fluorescent = real_fluorescent.to(self.device)
            masks = masks.to(self.device)
            
            # Generate random noise for generator
            noise = torch.randn(masks.size(0), 100, device=self.device)
            
            # Generate fluorescent images
            fake_fluorescent = self.generator(noise, masks)
            
            # Create grayscale overlay
            masks_norm = (masks + 1.0) / 2.0
            fake_norm = (fake_fluorescent + 1.0) / 2.0
            overlay = (masks_norm + fake_norm) / 2.0
            
            # Convert to numpy for display
            real_np = ((real_fluorescent.cpu() + 1) / 2).clamp(0, 1).numpy()
            masks_np = ((masks.cpu() + 1) / 2).clamp(0, 1).numpy()
            fake_np = ((fake_fluorescent.cpu() + 1) / 2).clamp(0, 1).numpy()
            overlay_np = overlay.cpu().clamp(0, 1).numpy()
            
            # Create display
            fig, axes = plt.subplots(4, 8, figsize=(16, 8))
            fig.suptitle(f'Epoch {epoch+1} - Conditioning Results', fontsize=14)
            
            for i in range(8):
                # Row 1: Real fluorescent
                axes[0, i].imshow(real_np[i, 0], cmap='gray')
                axes[0, i].set_title('Real' if i == 0 else '')
                axes[0, i].axis('off')
                
                # Row 2: Masks
                axes[1, i].imshow(masks_np[i, 0], cmap='gray')
                axes[1, i].set_title('Masks' if i == 0 else '')
                axes[1, i].axis('off')
                
                # Row 3: Generated
                axes[2, i].imshow(fake_np[i, 0], cmap='gray')
                axes[2, i].set_title('Generated' if i == 0 else '')
                axes[2, i].axis('off')
                
                # Row 4: Overlay
                axes[3, i].imshow(overlay_np[i, 0], cmap='gray')
                axes[3, i].set_title('Overlay' if i == 0 else '')
                axes[3, i].axis('off')
            
            plt.tight_layout()
            
            # Force display in Colab
            try:
                from IPython.display import display
                display(fig)
            except ImportError:
                pass  # Not in Jupyter/Colab
            
            plt.show()
            plt.close(fig)  # Clean up memory
            print(f"Displayed samples for epoch {epoch+1}")
        
        self.generator.train()
    
    def save_models(self, output_dir, epoch):
        """Save model checkpoints"""
        checkpoint = {
            'epoch': epoch + 1,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'ada_probs': self.ada_probs,
            'ada_aug_p': self.ada.ada_aug_p
        }
        
        save_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
    
    def save_training_curves(self, output_dir):
        """Save training loss curves"""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses')
        plt.grid(True)
        
        # ADA probability
        plt.subplot(1, 3, 2)
        plt.plot(self.ada_probs, label='ADA Probability', color='green')
        plt.axhline(y=self.ada.ada_target, color='red', linestyle='--', label=f'Target ({self.ada.ada_target})')
        plt.xlabel('Epoch')
        plt.ylabel('ADA Probability')
        plt.legend()
        plt.title('Adaptive Discriminator Augmentation')
        plt.grid(True)
        
        # Loss ratio
        plt.subplot(1, 3, 3)
        if len(self.g_losses) > 0 and len(self.d_losses) > 0:
            loss_ratio = np.array(self.g_losses) / (np.array(self.d_losses) + 1e-8)
            plt.plot(loss_ratio, label='G_loss / D_loss', color='purple')
            plt.axhline(y=1.0, color='red', linestyle='--', label='Balanced (1.0)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Ratio')
            plt.legend()
            plt.title('Generator/Discriminator Balance')
            plt.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved training curves to {save_path}")
    
    def save_final_results(self, output_dir):
        """Save final training results and summary"""
        results = {
            'final_g_loss': self.g_losses[-1] if self.g_losses else None,
            'final_d_loss': self.d_losses[-1] if self.d_losses else None,
            'final_ada_prob': self.ada_probs[-1] if self.ada_probs else None,
            'total_epochs': len(self.g_losses),
            'ada_target': self.ada.ada_target,
            'lr_g': self.lr_g,
            'lr_d': self.lr_d,
            'device': str(self.device),
            'model_type': 'simple' if self.use_simple_models else 'full'
        }
        
        with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Final Results:")
        print(f"  Generator Loss: {results['final_g_loss']:.6f}")
        print(f"  Discriminator Loss: {results['final_d_loss']:.6f}")
        print(f"  ADA Probability: {results['final_ada_prob']:.4f}")
        print(f"  Training completed successfully!")


def main():
    """Main training function for unpaired conditional generation"""
    parser = argparse.ArgumentParser(description='Train Conditional GAN with ADA and Structural Loss')
    parser.add_argument('--fluorescent_dir', type=str, required=True,
                      help='Directory containing real fluorescent images (for discriminator)')
    parser.add_argument('--mask_dir', type=str, required=True,
                      help='Directory containing mask images (conditions)')
    parser.add_argument('--output_dir', type=str, default='./outputs_ada',
                      help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size')
    parser.add_argument('--lr_g', type=float, default=0.0002,
                      help='Generator learning rate (conservative for stability)')
    parser.add_argument('--lr_d', type=float, default=0.0002,
                      help='Discriminator learning rate (conservative for stability)')
    parser.add_argument('--latent_dim', type=int, default=100,
                      help='Latent dimension')
    parser.add_argument('--image_size', type=int, default=256,
                      help='Image size')
    parser.add_argument('--ada_target', type=float, default=0.1,
                      help='ADA target accuracy (very low for maximum conditioning learning)')
    parser.add_argument('--ada_update', type=float, default=0.05,
                      help='ADA update step size (CellSynthesis default)')
    parser.add_argument('--use_simple_models', action='store_true',
                      help='Use simplified models')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create trainer for unpaired conditional generation
    trainer = ConditionalGANTrainer(
        fluorescent_dir=args.fluorescent_dir,  # Real fluorescent images
        mask_dir=args.mask_dir,                # Distance masks as conditions
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        device=args.device,
        use_simple_models=args.use_simple_models,
        ada_target=args.ada_target,
        ada_update=args.ada_update
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_interval=10,  # Save every 10 epochs
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
