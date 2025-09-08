"""
Conditional GAN Training Script for Fluorescent Microscopy Images
Inspired by CellSynthesis: https://github.com/stegmaierj/CellSynthesis

This script trains a conditional GAN to generate fluorescent cell images
conditioned on binary cell membrane segmentation masks using PyTorch Lightning
with Adaptive Discriminator Augmentation (ADA) for stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


class ConditionalGANTrainer:
    """
    Conditional GAN Trainer class for training with paired fluorescent images and masks
    """
    
    def __init__(self, fluorescent_dir, mask_dir, latent_dim=100, image_size=256, lr=0.0002, device=None, use_simple_models=False):
        """
        Initialize the conditional GAN trainer
        
        Args:
            fluorescent_dir (str): Directory containing fluorescent TIF images
            mask_dir (str): Directory containing binary mask TIF images
            latent_dim (int): Dimension of noise vector
            image_size (int): Size of generated images (assumed square)
            lr (float): Learning rate
            device (str): Device to use for training
            use_simple_models (bool): Whether to use simplified models
        """
        self.fluorescent_dir = fluorescent_dir
        self.mask_dir = mask_dir
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.lr = lr
        self.use_simple_models = use_simple_models
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize dataset
        try:
            self.dataset = ConditionalImageDataset(fluorescent_dir, mask_dir, image_size)
        except ValueError as e:
            print(f"Error with separate directories: {e}")
            print("Trying single directory approach...")
            # Fallback to single directory if separate dirs don't work
            self.dataset = SingleDirectoryConditionalDataset(fluorescent_dir, image_size=image_size)
        
        # Print dataset info
        dataset_info = self.dataset.get_dataset_info()
        print(f"Dataset Info:")
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")
        
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
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizers with different learning rates for balanced training
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr * 2, betas=(0.5, 0.999))  # Higher LR for G
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=20, gamma=0.8)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=20, gamma=0.8)
        
        # Training counters for balanced training
        self.g_train_freq = 1
        self.d_train_freq = 1
        
        # Training history
        self.G_losses = []
        self.D_losses = []
        self.D_real_acc = []
        self.D_fake_acc = []
        
        print("Conditional GAN initialized successfully!")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"Training image pairs available: {len(self.dataset)}")
    
    def _init_weights(self):
        """Initialize network weights"""
        def init_func(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        
        self.generator.apply(init_func)
        self.discriminator.apply(init_func)
    
    def generate_fake_data(self, batch_size, masks):
        """
        Generate fake fluorescent images conditioned on masks
        
        Args:
            batch_size (int): Number of samples to generate
            masks (torch.Tensor): Binary masks to condition on
            
        Returns:
            torch.Tensor: Generated fake fluorescent images
        """
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.generator(noise, masks)
        return fake_data
    
    def train_step(self, real_fluorescent, real_masks, batch_idx):
        """
        Perform one training step with real fluorescent images and masks
        
        Args:
            real_fluorescent (torch.Tensor): Real fluorescent images
            real_masks (torch.Tensor): Corresponding binary masks
            batch_idx (int): Current batch index
            
        Returns:
            tuple: (discriminator_loss, generator_loss, d_real_acc, d_fake_acc)
        """
        batch_size = real_fluorescent.size(0)
        
        # Labels with label smoothing
        real_label = 0.9
        fake_label = 0.1
        
        # Move data to device
        real_fluorescent = real_fluorescent.to(self.device)
        real_masks = real_masks.to(self.device)
        
        # Generate fake fluorescent images conditioned on real masks
        fake_fluorescent = self.generate_fake_data(batch_size, real_masks)
        
        # ============================================
        # Train Discriminator
        # ============================================
        d_loss = torch.tensor(0.0)
        d_real_acc = 0.0
        d_fake_acc = 0.0
        
        if batch_idx % self.d_train_freq == 0:
            self.discriminator.zero_grad()
            
            # Real data (fluorescent + mask)
            real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
            real_output = self.discriminator(real_fluorescent, real_masks)
            
            # Clamp discriminator output to prevent rounding errors
            real_output = torch.clamp(real_output, 1e-7, 1-1e-7)
            
            # Handle patch-based or simple discriminator output
            if len(real_output.shape) > 2:  # Patch-based
                real_output = real_output.mean(dim=[2, 3]).view(-1)
            else:
                real_output = real_output.view(-1)
            
            d_loss_real = self.criterion(real_output, real_labels)
            d_loss_real.backward()
            
            # Calculate accuracy on real data
            d_real_acc = ((torch.sigmoid(real_output) > 0.5).float() == (real_labels > 0.5).float()).float().mean().item()
            
            # Fake data (generated fluorescent + real mask)
            fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=self.device)
            fake_output = self.discriminator(fake_fluorescent.detach(), real_masks)
            
            # Clamp discriminator output to prevent rounding errors
            fake_output = torch.clamp(fake_output, 1e-7, 1-1e-7)
            
            # Handle patch-based or simple discriminator output
            if len(fake_output.shape) > 2:  # Patch-based
                fake_output = fake_output.mean(dim=[2, 3]).view(-1)
            else:
                fake_output = fake_output.view(-1)
            
            d_loss_fake = self.criterion(fake_output, fake_labels)
            d_loss_fake.backward()
            
            # Calculate accuracy on fake data
            d_fake_acc = ((torch.sigmoid(fake_output) > 0.5).float() == (fake_labels > 0.5).float()).float().mean().item()
            
            d_loss = d_loss_real + d_loss_fake
            self.optimizer_D.step()
        
        # ============================================
        # Train Generator
        # ============================================
        g_loss = torch.tensor(0.0)
        
        if batch_idx % self.g_train_freq == 0:
            self.generator.zero_grad()
            
            # Generate new fake data for generator training
            fake_fluorescent_for_g = self.generate_fake_data(batch_size, real_masks)
            
            # Try to fool discriminator
            fake_output_for_g = self.discriminator(fake_fluorescent_for_g, real_masks)
            
            # Clamp discriminator output to prevent rounding errors
            fake_output_for_g = torch.clamp(fake_output_for_g, 1e-7, 1-1e-7)
            
            # Handle patch-based or simple discriminator output
            if len(fake_output_for_g.shape) > 2:  # Patch-based
                fake_output_for_g = fake_output_for_g.mean(dim=[2, 3]).view(-1)
            else:
                fake_output_for_g = fake_output_for_g.view(-1)
            
            # Use real labels (want discriminator to think fake is real)
            real_labels_for_g = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
            g_loss = self.criterion(fake_output_for_g, real_labels_for_g)
            
            g_loss.backward()
            self.optimizer_G.step()
        
        # Adaptive training frequency
        self._adjust_training_frequency(d_real_acc, d_fake_acc)
        
        return d_loss.item(), g_loss.item(), d_real_acc, d_fake_acc
    
    def _adjust_training_frequency(self, d_real_acc, d_fake_acc):
        """Adjust training frequency based on discriminator performance"""
        if d_real_acc > 0.9 and d_fake_acc > 0.9:
            self.g_train_freq = 1
            self.d_train_freq = 2
        elif d_real_acc < 0.6 or d_fake_acc < 0.6:
            self.g_train_freq = 2
            self.d_train_freq = 1
        else:
            self.g_train_freq = 1
            self.d_train_freq = 1
    
    def train(self, num_epochs=100, batch_size=4, save_interval=5, quick_sample_freq=5):
        """
        Train the conditional GAN
        
        Args:
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size
            save_interval (int): Save interval for models and samples
            quick_sample_freq (int): Frequency of quick samples
        """
        print(f"\nStarting conditional GAN training for {num_epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Dataset size: {len(self.dataset)} image pairs")
        print(f"Quick samples every {quick_sample_freq} batch(es)")
        print("-" * 50)
        
        # Create data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        
        # Create output directories
        os.makedirs('cond_output/models', exist_ok=True)
        os.makedirs('cond_output/samples', exist_ok=True)
        os.makedirs('cond_output/batch_samples', exist_ok=True)
        
        # Fixed noise and mask for consistent sample generation
        fixed_noise = torch.randn(4, self.latent_dim, device=self.device)
        # Get a fixed mask from the dataset
        sample_fluorescent, sample_mask = self.dataset[0]
        fixed_mask = sample_mask.unsqueeze(0).repeat(4, 1, 1, 1).to(self.device)
        
        for epoch in range(num_epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_d_real_acc = 0.0
            epoch_d_fake_acc = 0.0
            num_batches = 0
            
            for batch_idx, (real_fluorescent, real_masks) in enumerate(dataloader):
                d_loss, g_loss, d_real_acc, d_fake_acc = self.train_step(real_fluorescent, real_masks, batch_idx)
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                epoch_d_real_acc += d_real_acc
                epoch_d_fake_acc += d_fake_acc
                num_batches += 1
                
                # Save quick samples
                if batch_idx % quick_sample_freq == 0:
                    self.save_quick_sample(epoch + 1, batch_idx, fixed_noise, fixed_mask)
                
                # Print batch progress
                if batch_idx % 5 == 0:
                    print(f"  Batch [{batch_idx}/{len(dataloader)}] - D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}, "
                          f"D_real_acc: {d_real_acc:.3f}, D_fake_acc: {d_fake_acc:.3f}")
            
            # Calculate averages
            avg_d_loss = epoch_d_loss / num_batches if num_batches > 0 else 0
            avg_g_loss = epoch_g_loss / num_batches if num_batches > 0 else 0
            avg_d_real_acc = epoch_d_real_acc / num_batches if num_batches > 0 else 0
            avg_d_fake_acc = epoch_d_fake_acc / num_batches if num_batches > 0 else 0
            
            self.D_losses.append(avg_d_loss)
            self.G_losses.append(avg_g_loss)
            self.D_real_acc.append(avg_d_real_acc)
            self.D_fake_acc.append(avg_d_fake_acc)
            
            # Step schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Print epoch progress
            print(f"Epoch [{epoch+1}/{num_epochs}] - Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}")
            print(f"  D_real_acc: {avg_d_real_acc:.3f}, D_fake_acc: {avg_d_fake_acc:.3f}")
            print(f"  G_LR: {self.scheduler_G.get_last_lr()[0]:.6f}, D_LR: {self.scheduler_D.get_last_lr()[0]:.6f}")
            
            # Save at intervals
            if (epoch + 1) % save_interval == 0 or epoch == 0:
                self.save_samples(epoch + 1)
                self.save_models(epoch + 1)
                self.plot_losses(epoch + 1)
        
        print("\nConditional GAN training completed!")
        self.save_models(num_epochs, final=True)
    
    def save_quick_sample(self, epoch, batch_idx, fixed_noise, fixed_mask):
        """Save a quick conditional sample"""
        self.generator.eval()
        with torch.no_grad():
            fake_sample = self.generator(fixed_noise[:1], fixed_mask[:1])
            
            # Create comparison figure
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Original mask
            mask_img = fixed_mask[0].cpu().squeeze().numpy()
            axes[0].imshow(mask_img, cmap='gray')
            axes[0].set_title('Input Mask')
            axes[0].axis('off')
            
            # Generated fluorescent
            fake_img = fake_sample[0].cpu().squeeze().numpy()
            fake_img = (fake_img + 1) / 2  # Denormalize
            axes[1].imshow(fake_img)
            axes[1].set_title('Generated Fluorescent')
            axes[1].axis('off')
            
            # Overlay
            overlay = np.stack([mask_img, fake_img, np.zeros_like(mask_img)], axis=-1)
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.suptitle(f'Conditional Generation - E{epoch:02d}_B{batch_idx:03d}')
            plt.tight_layout()
            
            filename = f'cond_output/batch_samples/epoch_{epoch:02d}_batch_{batch_idx:03d}.png'
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close()
        
        self.generator.train()
    
    def save_samples(self, epoch):
        """Save comprehensive sample comparisons"""
        self.generator.eval()
        with torch.no_grad():
            # Get real samples
            fluorescent_samples, mask_samples = self.dataset.get_sample_pairs(4)
            
            if fluorescent_samples:
                real_fluorescent = torch.cat(fluorescent_samples, dim=0).to(self.device)
                real_masks = torch.cat(mask_samples, dim=0).to(self.device)
                
                # Generate fake fluorescent images using real masks
                noise = torch.randn(4, self.latent_dim, device=self.device)
                fake_fluorescent = self.generator(noise, real_masks)
                
                # Create comparison grid
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                
                for i in range(4):
                    # Real masks
                    mask_img = real_masks[i].cpu().squeeze().numpy()
                    axes[0, i].imshow(mask_img, cmap='gray')
                    axes[0, i].set_title(f'Mask {i+1}')
                    axes[0, i].axis('off')
                    
                    # Real fluorescent
                    real_img = real_fluorescent[i].cpu().squeeze().numpy()
                    real_img = (real_img + 1) / 2
                    axes[1, i].imshow(real_img, cmap='green')
                    axes[1, i].set_title(f'Real Fluor {i+1}')
                    axes[1, i].axis('off')
                    
                    # Generated fluorescent
                    fake_img = fake_fluorescent[i].cpu().squeeze().numpy()
                    fake_img = (fake_img + 1) / 2
                    axes[2, i].imshow(fake_img, cmap='green')
                    axes[2, i].set_title(f'Generated Fluor {i+1}')
                    axes[2, i].axis('off')
                
                plt.suptitle(f'Conditional GAN Results - Epoch {epoch}')
                plt.tight_layout()
                plt.savefig(f'cond_output/samples/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        self.generator.train()
        print(f"Conditional samples saved for epoch {epoch}")
    
    def save_models(self, epoch, final=False):
        """Save model checkpoints"""
        if final:
            torch.save(self.generator.state_dict(), 'cond_output/models/conditional_generator_final.pth')
            torch.save(self.discriminator.state_dict(), 'cond_output/models/conditional_discriminator_final.pth')
            print("Final conditional models saved!")
        else:
            torch.save(self.generator.state_dict(), f'cond_output/models/conditional_generator_epoch_{epoch:03d}.pth')
            torch.save(self.discriminator.state_dict(), f'cond_output/models/conditional_discriminator_epoch_{epoch:03d}.pth')
    
    def plot_losses(self, epoch):
        """Plot training metrics for conditional GAN"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        ax1.plot(self.G_losses, label='Generator Loss', color='blue')
        ax1.plot(self.D_losses, label='Discriminator Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Conditional GAN Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot discriminator accuracies
        ax2.plot(self.D_real_acc, label='Real Data Accuracy', color='green')
        ax2.plot(self.D_fake_acc, label='Fake Data Accuracy', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Discriminator Accuracies')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Plot generator loss trend
        if len(self.G_losses) > 5:
            window = min(5, len(self.G_losses))
            g_loss_smooth = np.convolve(self.G_losses, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(self.G_losses)), g_loss_smooth, label='G Loss (smoothed)', color='blue')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Smoothed Loss')
            ax3.set_title('Generator Loss Trend')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot loss ratio
        if len(self.G_losses) > 0 and len(self.D_losses) > 0:
            loss_ratios = [g/(d+1e-8) for g, d in zip(self.G_losses, self.D_losses)]
            ax4.plot(loss_ratios, label='G_loss / D_loss', color='purple')
            ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Balanced (ratio=1)')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Ratio')
            ax4.set_title('Training Balance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'cond_output/samples/training_metrics_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()


def get_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def main():
    """Main training function with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Conditional GAN')
    parser.add_argument('--data_dir', type=str, default='/Users/edwheeler/cond_GAN/CellSimul/CellSimul/data',
                       help='Base directory containing data')
    parser.add_argument('--output_dir', type=str, default='/Users/edwheeler/cond_GAN/CellSimul/CellSimul/outputs',
                       help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/mps/cpu/auto)')
    parser.add_argument('--model_type', type=str, default='complex', choices=['simple', 'complex'],
                       help='Type of model to use')
    parser.add_argument('--save_frequency', type=int, default=5, help='Generate sample images every N epochs (models saved only at end)')
    parser.add_argument('--paired', action='store_true', help='Use paired training data')
    parser.add_argument('--fluorescence_dir', type=str, default='fluorescence_rescaled',
                       help='Directory name for fluorescence images (for unpaired training)')
    parser.add_argument('--distance_dir', type=str, default='distance_masks_rescaled', 
                       help='Directory name for distance masks (for unpaired training)')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to use for training (useful for quick tests)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Choose dataset based on training type
    if args.paired:
        print("Using paired training data...")
        # For paired data, use the old-style trainer configuration
        fluorescence_path = os.path.join(args.data_dir, args.fluorescence_dir)
        distance_path = os.path.join(args.data_dir, args.distance_dir)
        
        print(f"Fluorescence images: {fluorescence_path}")
        print(f"Distance masks: {distance_path}")
        
        # Initialize trainer with old approach
        trainer = ConditionalGANTrainer(
            fluorescent_dir=fluorescence_path,
            mask_dir=distance_path,
            latent_dim=100,
            image_size=256,
            lr=args.learning_rate,
            use_simple_models=(args.model_type == 'simple')
        )
        
        print(f"Starting paired training with {args.model_type} models...")
        print(f"Batch size: {args.batch_size}, Epochs: {args.num_epochs}")
        
        # Train using old approach
        trainer.train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            save_interval=args.save_frequency,
            quick_sample_freq=5
        )
        
    else:
        print("Using unpaired training data...")
        # Use unpaired dataset for real fluorescent + synthetic masks
        fluorescence_path = os.path.join(args.data_dir, args.fluorescence_dir)
        distance_path = os.path.join(args.data_dir, args.distance_dir)
        
        print(f"Fluorescence images: {fluorescence_path}")
        print(f"Distance masks: {distance_path}")
        
        if not os.path.exists(fluorescence_path):
            print(f"Error: Fluorescence directory '{fluorescence_path}' not found!")
            return
        if not os.path.exists(distance_path):
            print(f"Error: Distance masks directory '{distance_path}' not found!")
            return
        
        # Create unpaired dataset
        try:
            dataset = UnpairedConditionalImageDataset(
                fluorescent_dir=fluorescence_path,
                mask_dir=distance_path,
                image_size=256,
                max_images=args.max_images
            )
            print(f"Unpaired dataset size: {len(dataset)}")
            
            # Test the dataset
            print("Testing dataset...")
            sample_fluor, sample_mask = dataset[0]
            print(f"Sample fluorescent shape: {sample_fluor.shape}, range: [{sample_fluor.min():.3f}, {sample_fluor.max():.3f}]")
            print(f"Sample mask shape: {sample_mask.shape}, range: [{sample_mask.min():.3f}, {sample_mask.max():.3f}]")
            
            # Create data loader
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            
            # Initialize models based on type
            if args.model_type == 'simple':
                generator = SimpleConditionalGenerator().to(device)
                discriminator = SimpleConditionalDiscriminator().to(device)
            else:
                generator = ConditionalGenerator().to(device)
                discriminator = ConditionalDiscriminator().to(device)
            
            print(f"Initialized {args.model_type} models on {device}")
            
            # Initialize optimizers with unbalanced learning rates to prevent discriminator dominance
            g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate * 0.05, betas=(0.5, 0.999))  # Even more aggressive: 0.05x instead of 0.1x
            
            # Loss functions
            adversarial_criterion = nn.BCELoss()
            identity_criterion = nn.L1Loss()
            
            print(f"Starting unpaired training...")
            print(f"Batch size: {args.batch_size}, Epochs: {args.num_epochs}")
            print(f"Learning rate: {args.learning_rate} (G), {args.learning_rate * 0.05} (D)")  # Updated LR info
            
            # Progressive discriminator weakening scheduler
            d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.99)  # Decay D LR by 1% each epoch
            
            # Simple training loop for unpaired data following CellSynthesis structure
            for epoch in range(args.num_epochs):
                epoch_d_loss = 0.0
                epoch_g_loss = 0.0
                epoch_adv_loss = 0.0
                epoch_identity_loss = 0.0
                num_batches = 0
                
                for batch_idx, (real_fluorescent, condition_masks) in enumerate(dataloader):
                    real_fluorescent = real_fluorescent.to(device)
                    condition_masks = condition_masks.to(device)
                    batch_size = real_fluorescent.size(0)
                    
                    # ============================================
                    # Train Generator (improved feature learning)
                    # ============================================
                    g_optimizer.zero_grad()
                    
                    # Generate fake fluorescent images
                    z = torch.randn(batch_size, 100, device=device)
                    generated_fluorescent = generator(z, condition_masks)
                    
                    # ENHANCED CONDITIONING APPROACH v2.0
                    # Key insight: The current approach has negative correlation - we need to fix this
                    
                    # 1. WEAKER Identity Loss for Structure Preservation (further reduced weight)
                    real_fluorescent_as_condition = real_fluorescent
                    identity_fluorescent = generator(z, real_fluorescent_as_condition)
                    identity_loss = nn.L1Loss()(identity_fluorescent, real_fluorescent) * 3.0  # Reduced from 10.0
                    
                    # 2. DIRECT CORRELATION ENFORCEMENT
                    # The problem: we're getting negative correlation, so let's directly enforce positive correlation
                    mask_flat = condition_masks.view(batch_size, -1)
                    gen_flat = generated_fluorescent.view(batch_size, -1)
                    
                    # Normalize to zero mean for correlation calculation
                    mask_centered = mask_flat - mask_flat.mean(dim=1, keepdim=True)
                    gen_centered = gen_flat - gen_flat.mean(dim=1, keepdim=True)
                    
                    # Pearson correlation coefficient
                    mask_std = torch.sqrt(torch.sum(mask_centered ** 2, dim=1) + 1e-8)
                    gen_std = torch.sqrt(torch.sum(gen_centered ** 2, dim=1) + 1e-8)
                    correlation = torch.sum(mask_centered * gen_centered, dim=1) / (mask_std * gen_std + 1e-8)
                    
                    # Maximize positive correlation (penalize negative correlation more gently)
                    correlation_loss = torch.mean(torch.clamp(1.0 - correlation, min=0.0)) * 5.0  # Reduced from 15.0
                    
                    # 3. INTENSITY RATIO ENFORCEMENT
                    # High mask regions should have significantly higher intensity than low mask regions
                    mask_norm = (condition_masks + 1.0) / 2.0  # [0,1] range
                    gen_norm = (generated_fluorescent + 1.0) / 2.0  # [0,1] range
                    
                    # Define high/low regions based on mask quantiles
                    mask_high_threshold = torch.quantile(mask_norm.view(batch_size, -1), 0.75, dim=1).view(batch_size, 1, 1, 1)
                    mask_low_threshold = torch.quantile(mask_norm.view(batch_size, -1), 0.25, dim=1).view(batch_size, 1, 1, 1)
                    
                    high_mask_regions = (mask_norm >= mask_high_threshold).float()
                    low_mask_regions = (mask_norm <= mask_low_threshold).float()
                    
                    # Calculate mean intensities
                    high_intensity = torch.sum(gen_norm * high_mask_regions, dim=[1,2,3]) / (torch.sum(high_mask_regions, dim=[1,2,3]) + 1e-8)
                    low_intensity = torch.sum(gen_norm * low_mask_regions, dim=[1,2,3]) / (torch.sum(low_mask_regions, dim=[1,2,3]) + 1e-8)
                    
                    # Enforce high_intensity > low_intensity with margin (gentler)
                    intensity_ratio_loss = torch.mean(F.relu(low_intensity - high_intensity + 0.2)) * 2.0  # Reduced from 5.0
                    
                    # 4. WEAKER DIRECT INTENSITY MAPPING
                    # Reduce direct mapping to allow more creative generation
                    direct_mapping_loss = nn.MSELoss()(gen_norm, mask_norm) * 1.0  # Reduced from 3.0
                    
                    # 5. Feature matching loss (reduced weight to focus on conditioning)
                    with torch.no_grad():
                        real_features = discriminator(real_fluorescent, condition_masks)
                    fake_features = discriminator(generated_fluorescent, condition_masks)
                    
                    if real_features.dim() > 2:
                        real_features = real_features.view(batch_size, -1)
                        fake_features = fake_features.view(batch_size, -1)
                    
                    feature_matching_loss = nn.L1Loss()(fake_features, real_features) * 0.5  # Further reduced weight for more freedom
                    
                    # Combine conditioning losses
                    mask_consistency_loss = (correlation_loss + intensity_ratio_loss + direct_mapping_loss)
                    
                    # Adversarial loss - discriminator should think generated is real
                    d_output_fake = discriminator(generated_fluorescent, condition_masks)
                    
                    # Clamp discriminator output to prevent rounding errors
                    d_output_fake = torch.clamp(d_output_fake, 1e-7, 1-1e-7)
                    
                    # Flatten discriminator output if needed
                    if d_output_fake.dim() > 2:
                        d_output_fake = d_output_fake.view(batch_size, -1).mean(dim=1)
                    
                    # Use label smoothing for more stable training
                    valid_labels = torch.full((batch_size,), 0.9, device=device)
                    adversarial_loss = adversarial_criterion(d_output_fake, valid_labels)
                    
                    # CELLSYNTHESIS-STYLE GENERATOR LOSS: Balance adversarial and conditioning
                    # Increase adversarial weight relative to conditioning for more natural generation
                    g_loss = (adversarial_loss * 2 + identity_loss + feature_matching_loss + mask_consistency_loss) / 5  # Give more weight to adversarial
                    g_loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    
                    g_optimizer.step()
                    
                    # ============================================
                    # Train Discriminator (with frequency control)
                    # ============================================
                    
                    # Calculate current d_loss for frequency control decision (without training)
                    with torch.no_grad():
                        # Quick discriminator evaluation for frequency control
                        d_output_real_check = discriminator(real_fluorescent, condition_masks)
                        d_output_fake_check = discriminator(generated_fluorescent.detach(), condition_masks)
                        
                        # Flatten if needed
                        if d_output_real_check.dim() > 2:
                            d_output_real_check = d_output_real_check.view(batch_size, -1).mean(dim=1)
                        if d_output_fake_check.dim() > 2:
                            d_output_fake_check = d_output_fake_check.view(batch_size, -1).mean(dim=1)
                        
                        # Calculate current d_loss for condition checking with same aggressive smoothing
                        real_labels_check = torch.full((batch_size,), 0.7, device=device)  # Match training labels
                        fake_labels_check = torch.full((batch_size,), 0.3, device=device)  # Match training labels
                        d_real_loss_check = adversarial_criterion(torch.clamp(d_output_real_check, 1e-7, 1-1e-7), real_labels_check)
                        d_fake_loss_check = adversarial_criterion(torch.clamp(d_output_fake_check, 1e-7, 1-1e-7), fake_labels_check)
                        current_d_loss = (d_real_loss_check + d_fake_loss_check) / 2
                    
                    # Only train discriminator if it's not too strong
                    # Use current_d_loss instead of stale d_loss from previous batch
                    train_discriminator = True
                    if batch_idx > 5:  # Start control earlier
                        if current_d_loss.item() < 1.2:  # Even more aggressive threshold: 1.2 to target current 0.6 equilibrium
                            train_discriminator = (batch_idx % 20 == 0)  # Train D only every 20th iteration for stronger control
                    
                    if train_discriminator:
                        d_optimizer.zero_grad()
                        
                        # Add more aggressive noise to discriminator inputs to reduce overconfidence
                        noise_std = 0.1  # Increased from 0.05
                        real_fluorescent_noisy = real_fluorescent + torch.randn_like(real_fluorescent) * noise_std
                        generated_fluorescent_noisy = generated_fluorescent.detach() + torch.randn_like(generated_fluorescent) * noise_std
                        
                        # Set discriminator to training mode with dropout
                        discriminator.train()
                        
                        # Real samples - discriminator should output high values
                        d_output_real = discriminator(real_fluorescent_noisy, condition_masks)
                        
                        # Clamp discriminator output to prevent rounding errors
                        d_output_real = torch.clamp(d_output_real, 1e-7, 1-1e-7)
                        
                        # Flatten discriminator output if needed
                        if d_output_real.dim() > 2:
                            d_output_real = d_output_real.view(batch_size, -1).mean(dim=1)
                        
                        # Use even more aggressive label smoothing: real labels = 0.7 instead of 0.8
                        real_labels = torch.full((batch_size,), 0.7, device=device)
                        d_real_loss = adversarial_criterion(d_output_real, real_labels)
                        
                        # Fake samples - discriminator should output low values
                        d_output_fake_for_d = discriminator(generated_fluorescent_noisy, condition_masks)
                        
                        # Clamp discriminator output to prevent rounding errors
                        d_output_fake_for_d = torch.clamp(d_output_fake_for_d, 1e-7, 1-1e-7)
                        
                        # Flatten discriminator output if needed
                        if d_output_fake_for_d.dim() > 2:
                            d_output_fake_for_d = d_output_fake_for_d.view(batch_size, -1).mean(dim=1)
                        
                        # Use even more aggressive label smoothing: fake labels = 0.3 instead of 0.2
                        fake_labels = torch.full((batch_size,), 0.3, device=device)
                        d_fake_loss = adversarial_criterion(d_output_fake_for_d, fake_labels)
                        
                        # Combined discriminator loss
                        d_loss = (d_real_loss + d_fake_loss) / 2
                        d_loss.backward()
                        
                        # Gradient clipping to prevent discriminator from becoming too strong
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)  # Reduced clipping
                        
                        d_optimizer.step()
                    
                    # Accumulate losses (only accumulate d_loss if discriminator was trained)
                    if train_discriminator:
                        epoch_d_loss += d_loss.item()
                    epoch_g_loss += g_loss.item()
                    epoch_adv_loss += adversarial_loss.item()
                    epoch_identity_loss += identity_loss.item()  # Track identity loss
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
                        # Show current d_loss even when training is skipped
                        d_loss_str = f"{d_loss.item():.4f}" if train_discriminator else f"{current_d_loss.item():.4f}"
                        train_status = " [D_SKIP]" if not train_discriminator else ""
                        print(f"  Epoch [{epoch+1}/{args.num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                              f"D_loss: {d_loss_str}, G_loss: {g_loss.item():.4f}, "
                              f"Adv_loss: {adversarial_loss.item():.4f}, Feature_loss: {feature_matching_loss.item():.4f}, "
                              f"Identity_loss: {identity_loss.item():.4f}, Correlation_loss: {correlation_loss.item():.4f}, "
                              f"Intensity_ratio: {intensity_ratio_loss.item():.4f}, Direct_mapping: {direct_mapping_loss.item():.4f}{train_status}")
                
                # Epoch summary
                avg_d_loss = epoch_d_loss / num_batches
                avg_g_loss = epoch_g_loss / num_batches
                avg_adv_loss = epoch_adv_loss / num_batches
                avg_identity_loss = epoch_identity_loss / num_batches
                
                # Apply discriminator learning rate decay
                d_scheduler.step()
                current_d_lr = d_optimizer.param_groups[0]['lr']
                
                print(f"Epoch [{epoch+1}/{args.num_epochs}] Average D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
                print(f"  Identity_loss: {avg_identity_loss:.4f}, Adversarial_loss: {avg_adv_loss:.4f}, D_LR: {current_d_lr:.6f}")
                
                # Save models periodically every 100 epochs (overwrite)
                if (epoch + 1) % 100 == 0:
                    print(f"Saving models at epoch {epoch+1}...")
                    torch.save(generator.state_dict(), 
                              os.path.join(args.output_dir, 'unpaired_generator_checkpoint.pth'))
                    torch.save(discriminator.state_dict(), 
                              os.path.join(args.output_dir, 'unpaired_discriminator_checkpoint.pth'))
                    print(f"Checkpoint models saved at epoch {epoch+1}!")
                
                # Generate sample images for monitoring
                if (epoch + 1) % args.save_frequency == 0:
                    # Generate sample images for monitoring
                    generator.eval()
                    with torch.no_grad():
                        # Use a fixed number of samples
                        num_samples = min(4, batch_size)
                        sample_z = torch.randn(num_samples, 100, device=device)
                        sample_mask = condition_masks[:num_samples]  # Use first N masks from batch
                        sample_generated = generator(sample_z, sample_mask)
                        print(f"Generated sample shape: {sample_generated.shape}")
                        
                        # Print some statistics to monitor feature learning
                        real_mean = real_fluorescent.mean().item()
                        real_std = real_fluorescent.std().item()
                        gen_mean = sample_generated.mean().item()
                        gen_std = sample_generated.std().item()
                        mask_mean = sample_mask.mean().item()
                        
                        print(f"Real fluorescent - Mean: {real_mean:.3f}, Std: {real_std:.3f}")
                        print(f"Generated fluorescent - Mean: {gen_mean:.3f}, Std: {gen_std:.3f}")
                        print(f"Mask mean: {mask_mean:.3f} (normalized [-1,1] range)")
                        
                        # Check if there's correlation between mask and generated content
                        mask_flat = sample_mask.view(num_samples, -1)
                        gen_flat = sample_generated.view(num_samples, -1)
                        
                        # Compute correlation for each sample and take mean
                        correlations = []
                        for i in range(num_samples):
                            mask_sample = mask_flat[i]
                            gen_sample = gen_flat[i]
                            
                            # Normalize to [0,1] for correlation calculation
                            mask_norm = (mask_sample - mask_sample.min()) / (mask_sample.max() - mask_sample.min() + 1e-8)
                            gen_norm = (gen_sample + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
                            
                            # Calculate Pearson correlation
                            mask_centered = mask_norm - mask_norm.mean()
                            gen_centered = gen_norm - gen_norm.mean()
                            correlation = torch.sum(mask_centered * gen_centered) / (
                                torch.sqrt(torch.sum(mask_centered**2)) * torch.sqrt(torch.sum(gen_centered**2)) + 1e-8
                            )
                            correlations.append(correlation.item())
                        
                        avg_correlation = np.mean(correlations)
                        print(f"Mask-Generation correlation: {avg_correlation:.3f}")
                        
                        # Additional metric: intensity in high-distance areas vs low-distance areas
                        for i in range(min(2, num_samples)):  # Check first 2 samples
                            mask_sample = sample_mask[i].squeeze()
                            gen_sample = sample_generated[i].squeeze()
                            
                            # Normalize mask
                            mask_norm = (mask_sample - mask_sample.min()) / (mask_sample.max() - mask_sample.min() + 1e-8)
                            gen_norm = (gen_sample + 1.0) / 2.0
                            
                            # Create binary mask at 75th percentile
                            threshold = torch.quantile(mask_norm.flatten(), 0.75)
                            high_distance_mask = mask_norm > threshold
                            low_distance_mask = mask_norm <= threshold
                            
                            high_area_intensity = gen_norm[high_distance_mask].mean().item()
                            low_area_intensity = gen_norm[low_distance_mask].mean().item()
                            intensity_ratio = high_area_intensity / (low_area_intensity + 1e-8)
                            
                            print(f"Sample {i+1} - High/Low intensity ratio: {intensity_ratio:.3f} "
                                  f"(should be > 1.0 for good conditioning)")
                        
                        # Save sample generated images as TIF
                        import tifffile
                        
                        # Convert tensor to numpy and save each sample individually as TIF
                        sample_generated_np = sample_generated.cpu().numpy()
                        sample_mask_np = sample_mask.cpu().numpy()
                        
                        for i in range(num_samples):
                            # Save generated image (normalize from [-1,1] to [0,255])
                            gen_img = sample_generated_np[i, 0]  # Remove channel dimension
                            gen_img_normalized = ((gen_img + 1.0) * 127.5).astype(np.uint8)
                            gen_save_path = os.path.join(args.output_dir, f'unpaired_sample_epoch_{epoch+1}_img_{i+1}.tif')
                            tifffile.imwrite(gen_save_path, gen_img_normalized)
                            
                            # Save input mask (normalize from [-1,1] to [0,255])
                            mask_img = sample_mask_np[i, 0]  # Remove channel dimension  
                            mask_img_normalized = ((mask_img + 1.0) * 127.5).astype(np.uint8)
                            mask_save_path = os.path.join(args.output_dir, f'unpaired_mask_epoch_{epoch+1}_img_{i+1}.tif')
                            tifffile.imwrite(mask_save_path, mask_img_normalized)
                        
                        # Also create a combined visualization as TIF for easy viewing
                        import matplotlib.pyplot as plt
                        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*3, 6))
                        if num_samples == 1:
                            axes = axes.reshape(2, 1)
                        
                        for i in range(num_samples):
                            # Generated images
                            gen_img = sample_generated_np[i, 0]
                            axes[0, i].imshow(gen_img, cmap='gray', vmin=-1, vmax=1)
                            axes[0, i].set_title(f'Generated {i+1}')
                            axes[0, i].axis('off')
                            
                            # Input masks
                            mask_img = sample_mask_np[i, 0]
                            axes[1, i].imshow(mask_img, cmap='gray', vmin=-1, vmax=1)
                            axes[1, i].set_title(f'Input Mask {i+1}')
                            axes[1, i].axis('off')
                        
                        plt.tight_layout()
                        combined_save_path = os.path.join(args.output_dir, f'unpaired_combined_epoch_{epoch+1}.tif')
                        plt.savefig(combined_save_path, dpi=150, bbox_inches='tight', format='tiff')
                        plt.close()
                        
                        print(f"Sample images and input masks saved as TIF files at epoch {epoch+1}")
                    generator.train()
            
            # Save final models only at the very end
            print("Training completed! Saving final models...")
            torch.save(generator.state_dict(), 
                      os.path.join(args.output_dir, 'unpaired_generator_final.pth'))
            torch.save(discriminator.state_dict(), 
                      os.path.join(args.output_dir, 'unpaired_discriminator_final.pth'))
            print("Final models saved!")
            
            print("Unpaired training completed!")
            
        except Exception as e:
            print(f"Error during unpaired training: {e}")
            import traceback
            traceback.print_exc()
            return


if __name__ == "__main__":
    main()
