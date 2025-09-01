"""
Conditional GAN Training Script for Fluorescent Microscopy Images

This script trains a conditional GAN to generate fluorescent cell images
conditioned on binary cell membrane segmentation masks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from torch.utils.data import DataLoader

# Import our conditional models
from conditional_generator import ConditionalGenerator, SimpleConditionalGenerator
from conditional_discriminator import ConditionalDiscriminator, SimpleConditionalDiscriminator
from conditional_dataloader import ConditionalImageDataset, SingleDirectoryConditionalDataset


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
            axes[1].imshow(fake_img, cmap='green')
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


def main():
    """Main training function for conditional GAN"""
    # Configuration
    config = {
        'fluorescent_dir': '../data/fluorescent',  # Directory with fluorescent images
        'mask_dir': '../data/masks',               # Directory with binary masks
        'latent_dim': 100,
        'image_size': 256,
        'lr': 0.0001,
        'num_epochs': 100,
        'batch_size': 4,
        'save_interval': 5,
        'quick_sample_freq': 5,
        'use_simple_models': False  # Set to True for faster experimentation
    }
    
    print("Conditional GAN Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check if data directories exist
    if not os.path.exists(config['fluorescent_dir']):
        print(f"\nError: Fluorescent directory '{config['fluorescent_dir']}' not found!")
        print("Please create directories and place your files:")
        print("  - fluorescent images in ../data/fluorescent/")
        print("  - binary masks in ../data/masks/")
        return
    
    try:
        # Initialize and train conditional GAN
        gan = ConditionalGANTrainer(
            fluorescent_dir=config['fluorescent_dir'],
            mask_dir=config['mask_dir'],
            latent_dim=config['latent_dim'],
            image_size=config['image_size'],
            lr=config['lr'],
            use_simple_models=config['use_simple_models']
        )
        
        # Train the conditional GAN
        gan.train(
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            save_interval=config['save_interval'],
            quick_sample_freq=config['quick_sample_freq']
        )
        
        print("Conditional GAN training completed successfully!")
        print("Check the 'cond_output/' directory for:")
        print("  - Generated conditional samples")
        print("  - Saved model checkpoints")
        print("  - Training metrics and visualizations")
        
    except Exception as e:
        print(f"Error during conditional GAN training: {e}")
        print("Make sure you have paired fluorescent/mask TIF files and PyTorch installed.")


if __name__ == "__main__":
    main()
