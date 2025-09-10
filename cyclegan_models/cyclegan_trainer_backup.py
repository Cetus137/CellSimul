"""
CycleGAN Trainer for Mask-to-Fluorescent Translation

This trainer implements the complete CycleGAN training loop for translating
between distance masks and fluorescent images using unpaired data.

Key features:
- Bidirectional translation (Mask ↔ Fluorescent)
- Cycle consistency loss prevents memorization
- Identity loss for better preservation
- Learning rate scheduling
- Image buffer for discriminator stability
- Mixed precision training for speed
- M→F emphasis for improved quality
- Structural loss for mask-fluorescent correspondence
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import argparse
from collections import OrderedDict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from cyclegan_networks import CycleGANGenerator, CycleGANDiscriminator, CycleGANLoss, init_weights
from cond_models.unpaired_conditional_dataloader import UnpairedConditionalImageDataset


class StructuralLoss(nn.Module):
    """
    Structural loss to ensure generated fluorescent images follow mask structure.
    Combines edge consistency and spatial correlation losses.
    """
    def __init__(self):
        super(StructuralLoss, self).__init__()
        
        # Sobel edge detection kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def compute_edges(self, image):
        """Compute edge magnitude using Sobel filters"""
        # Apply Sobel filters
        edges_x = F.conv2d(image, self.sobel_x, padding=1)
        edges_y = F.conv2d(image, self.sobel_y, padding=1)
        
        # Compute edge magnitude
        edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)
        return edges
        
    def edge_consistency_loss(self, mask, generated_fluorescent):
        """Compute edge consistency between mask and generated fluorescent"""
        mask_edges = self.compute_edges(mask)
        fluor_edges = self.compute_edges(generated_fluorescent)
        
        # L1 loss between edge maps
        return F.l1_loss(mask_edges, fluor_edges)
        
    def spatial_correlation_loss(self, mask, generated_fluorescent):
        """Compute spatial correlation between mask and fluorescent images"""
        # Flatten spatial dimensions
        mask_flat = mask.view(mask.size(0), -1)
        fluor_flat = generated_fluorescent.view(generated_fluorescent.size(0), -1)
        
        # Center the data
        mask_centered = mask_flat - mask_flat.mean(dim=1, keepdim=True)
        fluor_centered = fluor_flat - fluor_flat.mean(dim=1, keepdim=True)
        
        # Compute correlation coefficient
        numerator = (mask_centered * fluor_centered).sum(dim=1)
        mask_std = torch.sqrt((mask_centered**2).sum(dim=1) + 1e-8)
        fluor_std = torch.sqrt((fluor_centered**2).sum(dim=1) + 1e-8)
        correlation = numerator / (mask_std * fluor_std + 1e-8)
        
        # We want high positive correlation, so minimize negative correlation
        return 1.0 - correlation.mean()
    
    def forward(self, mask, generated_fluorescent):
        """
        Compute structural loss between mask and generated fluorescent
        
        Args:
            mask: Input mask tensor
            generated_fluorescent: Generated fluorescent tensor from mask
        """
        edge_loss = self.edge_consistency_loss(mask, generated_fluorescent)
        correlation_loss = self.spatial_correlation_loss(mask, generated_fluorescent)
        
        # Combine losses (edge loss is more important for structure)
        total_loss = 2.0 * edge_loss + correlation_loss
        
        return {
            'total_loss': total_loss,
            'edge_loss': edge_loss,
            'correlation_loss': correlation_loss
        }


class ImageBuffer:
    """
    Image buffer for discriminator stability
    Stores previously generated images and returns them with 50% probability
    """
    def __init__(self, buffer_size=50):
        self.buffer_size = buffer_size
        self.num_imgs = 0
        self.images = []
    
    def push_and_pop(self, images):
        """Add images to buffer and return images for discriminator"""
        if self.buffer_size == 0:
            return images
        
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.buffer_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                if np.random.uniform() > 0.5:
                    random_id = np.random.randint(0, self.buffer_size)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        
        return_images = torch.cat(return_images, 0)
        return return_images


class CycleGANTrainer:
    """
    CycleGAN Trainer with optimizations for speed and quality
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.G_M2F = CycleGANGenerator(
            input_nc=config.input_nc, 
            output_nc=config.output_nc, 
            ngf=config.ngf, 
            n_residual_blocks=config.n_residual_blocks
        ).to(self.device)
        
        self.G_F2M = CycleGANGenerator(
            input_nc=config.output_nc, 
            output_nc=config.input_nc, 
            ngf=config.ngf, 
            n_residual_blocks=config.n_residual_blocks
        ).to(self.device)
        
        self.D_F = CycleGANDiscriminator(
            input_nc=config.output_nc, 
            ndf=config.ndf, 
            n_layers=config.n_discriminator_layers
        ).to(self.device)
        
        self.D_M = CycleGANDiscriminator(
            input_nc=config.input_nc, 
            ndf=config.ndf, 
            n_layers=config.n_discriminator_layers
        ).to(self.device)
        
        # Initialize weights
        init_weights(self.G_M2F)
        init_weights(self.G_F2M)
        init_weights(self.D_F)
        init_weights(self.D_M)
        
        # Loss functions
        self.loss_fn = CycleGANLoss(
            lambda_cycle=config.lambda_cycle,
            lambda_identity=config.lambda_identity
        )
        
        # Structural loss for mask-fluorescent correspondence
        self.structural_loss = StructuralLoss().to(self.device)
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            list(self.G_M2F.parameters()) + list(self.G_F2M.parameters()),
            lr=config.lr,
            betas=(config.beta1, 0.999)
        )
        
        self.optimizer_D_F = optim.Adam(
            self.D_F.parameters(),
            lr=config.lr,
            betas=(config.beta1, 0.999)
        )
        
        self.optimizer_D_M = optim.Adam(
            self.D_M.parameters(),
            lr=config.lr,
            betas=(config.beta1, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optimizer_G, 
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.n_epochs_decay) / config.n_epochs_decay
        )
        self.scheduler_D_F = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_F, 
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.n_epochs_decay) / config.n_epochs_decay
        )
        self.scheduler_D_M = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_M, 
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.n_epochs_decay) / config.n_epochs_decay
        )
        
        # Image buffers for discriminator stability
        self.fake_M_buffer = ImageBuffer(config.buffer_size)
        self.fake_F_buffer = ImageBuffer(config.buffer_size)
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training statistics
        self.current_epoch = 0
        self.losses = {
            'G_total': [],
            'G_M2F': [],
            'G_F2M': [],
            'cycle_M': [],
            'cycle_F': [],
            'identity_M': [],
            'identity_F': [],
            'D_F': [],
            'D_M': [],
            'structural': []
        }
        
    def load_checkpoint(self, checkpoint_path):
        """Load a pretrained checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model states
            self.G_M2F.load_state_dict(checkpoint['G_M2F_state_dict'])
            self.G_F2M.load_state_dict(checkpoint['G_F2M_state_dict'])
            self.D_F.load_state_dict(checkpoint['D_F_state_dict'])
            self.D_M.load_state_dict(checkpoint['D_M_state_dict'])
            
            # Load optimizers
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.optimizer_D_F.load_state_dict(checkpoint['optimizer_D_F_state_dict'])
            self.optimizer_D_M.load_state_dict(checkpoint['optimizer_D_M_state_dict'])
            
            # Load schedulers
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            self.scheduler_D_F.load_state_dict(checkpoint['scheduler_D_F_state_dict'])
            self.scheduler_D_M.load_state_dict(checkpoint['scheduler_D_M_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.losses = checkpoint.get('losses', self.losses)
            
            print(f"Checkpoint loaded successfully from epoch {self.current_epoch}")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def save_checkpoint(self, epoch, checkpoint_dir):
        """Save training checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'cyclegan_epoch_{epoch}.pth')
        
        torch.save({
            'epoch': epoch,
            'G_M2F_state_dict': self.G_M2F.state_dict(),
            'G_F2M_state_dict': self.G_F2M.state_dict(),
            'D_F_state_dict': self.D_F.state_dict(),
            'D_M_state_dict': self.D_M.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_F_state_dict': self.optimizer_D_F.state_dict(),
            'optimizer_D_M_state_dict': self.optimizer_D_M.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_F_state_dict': self.scheduler_D_F.state_dict(),
            'scheduler_D_M_state_dict': self.scheduler_D_M.state_dict(),
            'losses': self.losses,
            'config': self.config
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
    def create_summary_tiff(self, real_M, real_F, fake_F, fake_M, rec_M, rec_F, output_path):
        """Create a 6-row summary TIFF with individual normalization"""
        # Convert to numpy and normalize each image individually
        def normalize_individual(tensor):
            """Normalize each image in the batch individually to [0, 255]"""
            normalized = []
            for i in range(tensor.size(0)):
                img = tensor[i].detach().cpu().numpy().squeeze()
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min) * 255
                else:
                    img = np.zeros_like(img)
                normalized.append(img.astype(np.uint8))
            return normalized
        
        # Normalize all image sets
        real_M_norm = normalize_individual(real_M[:4])  # Take first 4 for display
        real_F_norm = normalize_individual(real_F[:4])
        fake_F_norm = normalize_individual(fake_F[:4])
        fake_M_norm = normalize_individual(fake_M[:4])
        rec_M_norm = normalize_individual(rec_M[:4])
        rec_F_norm = normalize_individual(rec_F[:4])
        
        # Create 6-row grid (each row shows 4 images)
        rows = [real_M_norm, fake_F_norm, real_F_norm, fake_M_norm, rec_M_norm, rec_F_norm]
        row_labels = ['Real_M', 'Fake_F', 'Real_F', 'Fake_M', 'Rec_M', 'Rec_F']
        
        # Stack images vertically
        grid_images = []
        for row_idx, (row_images, label) in enumerate(zip(rows, row_labels)):
            # Horizontally concatenate 4 images in this row
            row_concat = np.hstack(row_images)
            grid_images.append(row_concat)
        
        # Vertically concatenate all rows
        final_grid = np.vstack(grid_images)
        
        # Save as TIFF
        pil_image = Image.fromarray(final_grid, mode='L')  # Grayscale
        pil_image.save(output_path)
        print(f"Summary TIFF saved: {output_path}")
        
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with mixed precision and M→F emphasis"""
        self.G_M2F.train()
        self.G_F2M.train()
        self.D_F.train()
        self.D_M.train()
        
        epoch_losses = {
            'G_total': 0.0,
            'G_M2F': 0.0,
            'G_F2M': 0.0,
            'cycle_M': 0.0,
            'cycle_F': 0.0,
            'identity_M': 0.0,
            'identity_F': 0.0,
            'D_F': 0.0,
            'D_M': 0.0,
            'structural': 0.0
        }
        
        num_batches = len(dataloader)
        
        for batch_idx, (real_M, real_F) in enumerate(dataloader):
            real_M = real_M.to(self.device)
            real_F = real_F.to(self.device)
            
            # Mixed precision context
            if self.scaler:
                with autocast():
                    losses = self._train_step(real_M, real_F, epoch)
            else:
                losses = self._train_step(real_M, real_F, epoch)
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                      f"G_total: {losses['G_total']:.4f}, "
                      f"D_F: {losses['D_F']:.4f}, "
                      f"D_M: {losses['D_M']:.4f}, "
                      f"Structural: {losses['structural']:.4f}")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            self.losses[key].append(epoch_losses[key])
        
        return epoch_losses
    
    def _train_step(self, real_M, real_F, epoch):
        """Single training step with M→F emphasis"""
        # Generate fake images
        fake_F = self.G_M2F(real_M)
        fake_M = self.G_F2M(real_F)
        
        # Cycle consistency
        rec_M = self.G_F2M(fake_F)
        rec_F = self.G_M2F(fake_M)
        
        # Identity mapping
        identity_M = self.G_F2M(real_M)
        identity_F = self.G_M2F(real_F)
        
        # ============ Train Generators ============
        self.optimizer_G.zero_grad()
        
        if self.scaler:
            with autocast():
                # Adversarial loss
                pred_fake_F = self.D_F(fake_F)
                pred_fake_M = self.D_M(fake_M)
                
                loss_G_M2F = self.loss_fn.generator_loss(pred_fake_F)
                loss_G_F2M = self.loss_fn.generator_loss(pred_fake_M)
                
                # Cycle consistency loss
                loss_cycle_M = self.loss_fn.cycle_loss(real_M, rec_M)
                loss_cycle_F = self.loss_fn.cycle_loss(real_F, rec_F)
                
                # Identity loss
                loss_identity_M = self.loss_fn.identity_loss(real_M, identity_M)
                loss_identity_F = self.loss_fn.identity_loss(real_F, identity_F)
                
                # Structural loss for M→F translation
                structural_loss_dict = self.structural_loss(real_M, fake_F)
                loss_structural = structural_loss_dict['total_loss']
                
                # M→F emphasis: Weight M→F losses more heavily
                m2f_emphasis = getattr(self.config, 'm2f_emphasis_weight', 2.0)
                
                loss_G_total = (
                    m2f_emphasis * loss_G_M2F + loss_G_F2M +
                    m2f_emphasis * loss_cycle_F + loss_cycle_M +
                    m2f_emphasis * loss_identity_F + loss_identity_M +
                    getattr(self.config, 'structural_loss_weight', 1.0) * loss_structural
                )
            
            self.scaler.scale(loss_G_total).backward()
            self.scaler.step(self.optimizer_G)
            self.scaler.update()
        else:
            # Standard precision training
            pred_fake_F = self.D_F(fake_F)
            pred_fake_M = self.D_M(fake_M)
            
            loss_G_M2F = self.loss_fn.generator_loss(pred_fake_F)
            loss_G_F2M = self.loss_fn.generator_loss(pred_fake_M)
            
            loss_cycle_M = self.loss_fn.cycle_loss(real_M, rec_M)
            loss_cycle_F = self.loss_fn.cycle_loss(real_F, rec_F)
            
            loss_identity_M = self.loss_fn.identity_loss(real_M, identity_M)
            loss_identity_F = self.loss_fn.identity_loss(real_F, identity_F)
            
            structural_loss_dict = self.structural_loss(real_M, fake_F)
            loss_structural = structural_loss_dict['total_loss']
            
            # M→F emphasis
            m2f_emphasis = getattr(self.config, 'm2f_emphasis_weight', 2.0)
            
            loss_G_total = (
                m2f_emphasis * loss_G_M2F + loss_G_F2M +
                m2f_emphasis * loss_cycle_F + loss_cycle_M +
                m2f_emphasis * loss_identity_F + loss_identity_M +
                getattr(self.config, 'structural_loss_weight', 1.0) * loss_structural
            )
            
            loss_G_total.backward()
            self.optimizer_G.step()
        
        # ============ Train Discriminator F ============
        self.optimizer_D_F.zero_grad()
        
        # Real images
        pred_real_F = self.D_F(real_F)
        loss_D_F_real = self.loss_fn.discriminator_loss(pred_real_F, True)
        
        # Fake images from buffer
        fake_F_from_buffer = self.fake_F_buffer.push_and_pop(fake_F.detach())
        pred_fake_F = self.D_F(fake_F_from_buffer)
        loss_D_F_fake = self.loss_fn.discriminator_loss(pred_fake_F, False)
        
        loss_D_F = (loss_D_F_real + loss_D_F_fake) * 0.5
        
        if self.scaler:
            self.scaler.scale(loss_D_F).backward()
            self.scaler.step(self.optimizer_D_F)
        else:
            loss_D_F.backward()
            self.optimizer_D_F.step()
        
        # ============ Train Discriminator M ============
        self.optimizer_D_M.zero_grad()
        
        # Real images
        pred_real_M = self.D_M(real_M)
        loss_D_M_real = self.loss_fn.discriminator_loss(pred_real_M, True)
        
        # Fake images from buffer
        fake_M_from_buffer = self.fake_M_buffer.push_and_pop(fake_M.detach())
        pred_fake_M = self.D_M(fake_M_from_buffer)
        loss_D_M_fake = self.loss_fn.discriminator_loss(pred_fake_M, False)
        
        loss_D_M = (loss_D_M_real + loss_D_M_fake) * 0.5
        
        if self.scaler:
            self.scaler.scale(loss_D_M).backward()
            self.scaler.step(self.optimizer_D_M)
        else:
            loss_D_M.backward()
            self.optimizer_D_M.step()
        
        return {
            'G_total': loss_G_total.item(),
            'G_M2F': loss_G_M2F.item(),
            'G_F2M': loss_G_F2M.item(),
            'cycle_M': loss_cycle_M.item(),
            'cycle_F': loss_cycle_F.item(),
            'identity_M': loss_identity_M.item(),
            'identity_F': loss_identity_F.item(),
            'D_F': loss_D_F.item(),
            'D_M': loss_D_M.item(),
            'structural': loss_structural.item()
        }
        
    def train(self, dataloader, num_epochs, save_interval=10, sample_interval=5, output_dir='outputs'):
        """Main training loop"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"M→F emphasis weight: {getattr(self.config, 'm2f_emphasis_weight', 2.0)}")
        print(f"Structural loss weight: {getattr(self.config, 'structural_loss_weight', 1.0)}")
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            epoch_losses = self.train_epoch(dataloader, epoch)
            
            # Update learning rates
            self.scheduler_G.step()
            self.scheduler_D_F.step()
            self.scheduler_D_M.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Generator Total Loss: {epoch_losses['G_total']:.4f}")
            print(f"  M→F Generator Loss: {epoch_losses['G_M2F']:.4f}")
            print(f"  F→M Generator Loss: {epoch_losses['G_F2M']:.4f}")
            print(f"  Cycle M Loss: {epoch_losses['cycle_M']:.4f}")
            print(f"  Cycle F Loss: {epoch_losses['cycle_F']:.4f}")
            print(f"  Structural Loss: {epoch_losses['structural']:.4f}")
            print(f"  Discriminator F Loss: {epoch_losses['D_F']:.4f}")
            print(f"  Discriminator M Loss: {epoch_losses['D_M']:.4f}")
            
            # Save samples
            if epoch % sample_interval == 0:
                self.generate_samples(dataloader, epoch, output_dir)
            
            # Save checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch, os.path.join(output_dir, 'checkpoints'))
        
        print("Training completed!")
        
    def generate_samples(self, dataloader, epoch, output_dir):
        """Generate sample images"""
        self.G_M2F.eval()
        self.G_F2M.eval()
        
        with torch.no_grad():
            # Get a batch of real images
            real_M, real_F = next(iter(dataloader))
            real_M = real_M.to(self.device)
            real_F = real_F.to(self.device)
            
            # Generate fake images
            fake_F = self.G_M2F(real_M)
            fake_M = self.G_F2M(real_F)
            
            # Generate reconstructions
            rec_M = self.G_F2M(fake_F)
            rec_F = self.G_M2F(fake_M)
            
            # Create summary TIFF
            tiff_path = os.path.join(output_dir, 'samples', f'summary_epoch_{epoch}.tiff')
            self.create_summary_tiff(real_M, real_F, fake_F, fake_M, rec_M, rec_F, tiff_path)
        
        self.G_M2F.train()
        self.G_F2M.train()
    
    def plot_losses(self, output_dir):
        """Plot training losses"""
        epochs = range(len(self.losses['G_total']))
        
        plt.figure(figsize=(15, 10))
        
        # Generator losses
        plt.subplot(2, 3, 1)
        plt.plot(epochs, self.losses['G_total'], label='Total')
        plt.plot(epochs, self.losses['G_M2F'], label='M→F')
        plt.plot(epochs, self.losses['G_F2M'], label='F→M')
        plt.title('Generator Losses')
        plt.legend()
        
        # Cycle losses
        plt.subplot(2, 3, 2)
        plt.plot(epochs, self.losses['cycle_M'], label='Cycle M')
        plt.plot(epochs, self.losses['cycle_F'], label='Cycle F')
        plt.title('Cycle Consistency Losses')
        plt.legend()
        
        # Identity losses
        plt.subplot(2, 3, 3)
        plt.plot(epochs, self.losses['identity_M'], label='Identity M')
        plt.plot(epochs, self.losses['identity_F'], label='Identity F')
        plt.title('Identity Losses')
        plt.legend()
        
        # Discriminator losses
        plt.subplot(2, 3, 4)
        plt.plot(epochs, self.losses['D_F'], label='D_F')
        plt.plot(epochs, self.losses['D_M'], label='D_M')
        plt.title('Discriminator Losses')
        plt.legend()
        
        # Structural loss
        plt.subplot(2, 3, 5)
        plt.plot(epochs, self.losses['structural'], label='Structural')
        plt.title('Structural Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_losses.png'))
        plt.close()


def create_config():
    """Create configuration object"""
    parser = argparse.ArgumentParser(description='CycleGAN Training')
    
    # Model architecture
    parser.add_argument('--input_nc', type=int, default=1, help='Input channels')
    parser.add_argument('--output_nc', type=int, default=1, help='Output channels')
    parser.add_argument('--ngf', type=int, default=64, help='Generator filters')
    parser.add_argument('--ndf', type=int, default=64, help='Discriminator filters')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='ResNet blocks')
    parser.add_argument('--n_discriminator_layers', type=int, default=3, help='Discriminator layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='Epochs to decay LR')
    
    # Loss weights
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Cycle loss weight')
    parser.add_argument('--lambda_identity', type=float, default=5.0, help='Identity loss weight')
    parser.add_argument('--m2f_emphasis_weight', type=float, default=2.0, help='M→F emphasis weight')
    parser.add_argument('--structural_loss_weight', type=float, default=1.0, help='Structural loss weight')
    
    # Training optimizations
    parser.add_argument('--buffer_size', type=int, default=50, help='Image buffer size')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--fast_mode', action='store_true', help='Fast training mode')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, default='cyclegan_outputs', help='Output directory')
    parser.add_argument('--checkpoint_path', type=str, help='Pretrained checkpoint path')
    
    # Intervals
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval')
    parser.add_argument('--sample_interval', type=int, default=5, help='Sample interval')
    
    return parser.parse_args()


def main():
    config = create_config()
    
    # Fast mode adjustments
    if config.fast_mode:
        print("Fast mode enabled - increasing batch size and learning rate")
        config.batch_size = min(config.batch_size * 2, 16)
        config.lr *= 1.5
        config.mixed_precision = True
    
    # Create dataset
    print(f"Loading data from {config.data_dir}")
    dataset = UnpairedConditionalImageDataset(
        data_dir=config.data_dir,
        transform=None,  # Add transforms if needed
        mode='train'
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create trainer
    trainer = CycleGANTrainer(config)
    
    # Load pretrained checkpoint if provided
    if config.checkpoint_path:
        trainer.load_checkpoint(config.checkpoint_path)
    
    # Train
    trainer.train(
        dataloader=dataloader,
        num_epochs=config.n_epochs,
        save_interval=config.save_interval,
        sample_interval=config.sample_interval,
        output_dir=config.output_dir
    )
    
    # Plot losses
    trainer.plot_losses(config.output_dir)
    
    print(f"Training completed! Results saved to {config.output_dir}")


if __name__ == '__main__':
    main()
