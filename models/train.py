"""
GAN Training Script with Real TIF Images

This script defines and trains a GAN using the Generator and Discriminator models
for generating synthetic cell membrane images using real TIF files as training data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from torch.utils.data import DataLoader

# Import our models
from generator import Generator
from discriminator import Discriminator
from dataloader import TIFImageDataset


class GANTrainer:
    """
    GAN Trainer class for training with real TIF images
    """
    
    def __init__(self, data_dir, latent_dim=100, image_size=256, lr=0.0002, device=None):
        """
        Initialize the GAN trainer
        
        Args:
            data_dir (str): Directory containing TIF training images
            latent_dim (int): Dimension of noise vector
            image_size (int): Size of generated images (assumed square)
            lr (float): Learning rate
            device (str): Device to use for training
        """
        self.data_dir = data_dir
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.lr = lr
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize dataset
        self.dataset = TIFImageDataset(data_dir, image_size)
        
        # Print dataset info
        dataset_info = self.dataset.get_image_info()
        print(f"Dataset Info:")
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")
        
        # Initialize models
        self.generator = Generator(latent_dim=latent_dim).to(self.device)
        self.discriminator = Discriminator(in_channels=1, features=64).to(self.device)
        
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
        self.g_train_freq = 1  # Train generator every batch
        self.d_train_freq = 1  # Train discriminator every batch
        
        # Training history
        self.G_losses = []
        self.D_losses = []
        self.D_real_acc = []  # Track discriminator accuracy on real images
        self.D_fake_acc = []  # Track discriminator accuracy on fake images
        
        print("GAN initialized successfully!")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"Training images available: {len(self.dataset)}")
    
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
    
    def generate_fake_data(self, batch_size):
        """
        Generate fake data using random noise
        
        Args:
            batch_size (int): Number of samples to generate
            
        Returns:
            torch.Tensor: Generated fake images
        """
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.generator(noise)
        return fake_data
    
    def train_step(self, real_batch, batch_idx):
        """
        Perform one training step with real TIF images and balanced training
        
        Args:
            real_batch (torch.Tensor): Batch of real images from TIF files
            batch_idx (int): Current batch index for training balancing
            
        Returns:
            tuple: (discriminator_loss, generator_loss, d_real_acc, d_fake_acc)
        """
        batch_size = real_batch.size(0)
        
        # Labels with label smoothing for better training
        real_label = 0.9  # Label smoothing: use 0.9 instead of 1.0
        fake_label = 0.1  # Label smoothing: use 0.1 instead of 0.0
        
        # Move real data to device
        real_data = real_batch.to(self.device)
        
        # Generate fake data
        fake_data = self.generate_fake_data(batch_size)
        
        # ============================================
        # Train Discriminator (with frequency control)
        # ============================================
        d_loss = torch.tensor(0.0)
        d_real_acc = 0.0
        d_fake_acc = 0.0
        
        if batch_idx % self.d_train_freq == 0:
            self.discriminator.zero_grad()
            
            # Real data
            real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
            real_output = self.discriminator(real_data)
            
            # Handle patch-based discriminator output
            if len(real_output.shape) > 2:  # If output is patch-based (e.g., 7x7)
                # Take mean across spatial dimensions to get single value per sample
                real_output = real_output.mean(dim=[2, 3]).view(-1)
            else:
                real_output = real_output.view(-1)
                
            d_loss_real = self.criterion(real_output, real_labels)
            d_loss_real.backward()
            
            # Calculate accuracy on real data
            d_real_acc = ((torch.sigmoid(real_output) > 0.5).float() == (real_labels > 0.5).float()).float().mean().item()
            
            # Fake data
            fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=self.device)
            fake_output = self.discriminator(fake_data.detach())
            
            # Handle patch-based discriminator output
            if len(fake_output.shape) > 2:  # If output is patch-based (e.g., 7x7)
                # Take mean across spatial dimensions to get single value per sample
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
        # Train Generator (with frequency control and feature matching)
        # ============================================
        g_loss = torch.tensor(0.0)
        
        if batch_idx % self.g_train_freq == 0:
            self.generator.zero_grad()
            
            # Generate new fake data for generator training
            fake_data_for_g = self.generate_fake_data(batch_size)
            
            # Try to fool discriminator
            fake_output_for_g = self.discriminator(fake_data_for_g)
            
            # Handle patch-based discriminator output
            if len(fake_output_for_g.shape) > 2:  # If output is patch-based (e.g., 7x7)
                # Take mean across spatial dimensions to get single value per sample
                fake_output_for_g = fake_output_for_g.mean(dim=[2, 3]).view(-1)
            else:
                fake_output_for_g = fake_output_for_g.view(-1)
            
            # Use real labels (want discriminator to think fake is real)
            real_labels_for_g = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
            g_loss = self.criterion(fake_output_for_g, real_labels_for_g)
            
            # Add feature matching loss for better training
            if hasattr(self, 'feature_matching') and self.feature_matching:
                # Get intermediate features from discriminator
                real_features = self._get_discriminator_features(real_data)
                fake_features = self._get_discriminator_features(fake_data_for_g)
                
                # Feature matching loss
                feature_loss = nn.MSELoss()(fake_features, real_features.detach())
                g_loss = g_loss + 0.1 * feature_loss  # Small weight for feature matching
            
            g_loss.backward()
            self.optimizer_G.step()
        
        # Adaptive training frequency based on discriminator performance
        self._adjust_training_frequency(d_real_acc, d_fake_acc)
        
        return d_loss.item(), g_loss.item(), d_real_acc, d_fake_acc
    
    def _adjust_training_frequency(self, d_real_acc, d_fake_acc):
        """
        Adjust training frequency based on discriminator performance
        
        Args:
            d_real_acc (float): Discriminator accuracy on real data
            d_fake_acc (float): Discriminator accuracy on fake data
        """
        # If discriminator is too good (> 90% accuracy), train generator more frequently
        if d_real_acc > 0.9 and d_fake_acc > 0.9:
            self.g_train_freq = 1
            self.d_train_freq = 2  # Train discriminator less frequently
        # If discriminator is too bad (< 60% accuracy), train discriminator more frequently
        elif d_real_acc < 0.6 or d_fake_acc < 0.6:
            self.g_train_freq = 2  # Train generator less frequently
            self.d_train_freq = 1
        else:
            # Balanced training
            self.g_train_freq = 1
            self.d_train_freq = 1
    
    def _get_discriminator_features(self, x):
        """
        Extract intermediate features from discriminator for feature matching
        (This is a simplified version - you might want to modify discriminator to expose features)
        """
        # For now, return the final output as a feature
        # In a full implementation, you'd modify the discriminator to return intermediate features
        with torch.no_grad():
            features = self.discriminator(x)
            if len(features.shape) > 2:
                features = features.mean(dim=[2, 3])
        return features.detach()
    
    def train(self, num_epochs=100, batch_size=8, save_interval=10, quick_sample_freq=1):
        """
        Train the GAN with real TIF images
        
        Args:
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size
            save_interval (int): Save interval for models and samples
            quick_sample_freq (int): Frequency of quick samples (every N batches)
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Dataset size: {len(self.dataset)} images")
        print(f"Quick samples every {quick_sample_freq} batch(es)")
        print("-" * 50)
        
        # Create data loader
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            drop_last=True  # Drop last incomplete batch
        )
        
        # Create output directories
        os.makedirs('output/models', exist_ok=True)
        os.makedirs('output/samples', exist_ok=True)
        os.makedirs('output/batch_samples', exist_ok=True)  # New directory for frequent samples
        
        # Fixed noise for consistent sample generation
        fixed_noise = torch.randn(4, self.latent_dim, device=self.device)
        
        for epoch in range(num_epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_d_real_acc = 0.0
            epoch_d_fake_acc = 0.0
            num_batches = 0
            
            for batch_idx, real_batch in enumerate(dataloader):
                d_loss, g_loss, d_real_acc, d_fake_acc = self.train_step(real_batch, batch_idx)
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                epoch_d_real_acc += d_real_acc
                epoch_d_fake_acc += d_fake_acc
                num_batches += 1
                
                # Save example image after every N generator runs
                if batch_idx % quick_sample_freq == 0:
                    self.save_quick_sample(epoch + 1, batch_idx, fixed_noise)
                
                # Print batch progress occasionally
                if batch_idx % 10 == 0:
                    print(f"  Batch [{batch_idx}/{len(dataloader)}] - D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}, "
                          f"D_real_acc: {d_real_acc:.3f}, D_fake_acc: {d_fake_acc:.3f}")
            
            # Calculate average losses and accuracies
            avg_d_loss = epoch_d_loss / num_batches if num_batches > 0 else 0
            avg_g_loss = epoch_g_loss / num_batches if num_batches > 0 else 0
            avg_d_real_acc = epoch_d_real_acc / num_batches if num_batches > 0 else 0
            avg_d_fake_acc = epoch_d_fake_acc / num_batches if num_batches > 0 else 0
            
            self.D_losses.append(avg_d_loss)
            self.G_losses.append(avg_g_loss)
            self.D_real_acc.append(avg_d_real_acc)
            self.D_fake_acc.append(avg_d_fake_acc)
            
            # Step learning rate schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Print epoch progress
            print(f"Epoch [{epoch+1}/{num_epochs}] - Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}")
            print(f"  D_real_acc: {avg_d_real_acc:.3f}, D_fake_acc: {avg_d_fake_acc:.3f}")
            print(f"  G_LR: {self.scheduler_G.get_last_lr()[0]:.6f}, D_LR: {self.scheduler_D.get_last_lr()[0]:.6f}")
            
            # Save samples and models at intervals
            if (epoch + 1) % save_interval == 0 or epoch == 0:
                self.save_samples(epoch + 1)
                self.save_models(epoch + 1)
                self.plot_losses(epoch + 1)
        
        print("\nTraining completed!")
        self.save_models(num_epochs, final=True)
    
    def save_samples(self, epoch, num_samples=8):
        """Save sample generated images alongside real images for comparison"""
        self.generator.eval()
        with torch.no_grad():
            fake_data = self.generate_fake_data(num_samples)
            
            # Get real samples using the dataset's helper method
            real_samples = self.dataset.get_sample_images(num_samples)
            
            if real_samples:
                real_data = torch.cat(real_samples, dim=0)
            
            # Create comparison grid
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            
            # Plot fake images in top 2 rows
            for i in range(min(num_samples, 8)):
                row = i // 4
                col = i % 4
                img = fake_data[i].cpu().squeeze().numpy()
                # Denormalize from [-1, 1] to [0, 1]
                img = (img + 1) / 2
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].axis('off')
                axes[row, col].set_title(f'Generated {i+1}')
            
            # Plot real images in bottom 2 rows (if available)
            if real_samples:
                for i in range(min(len(real_samples), 8)):
                    row = 2 + i // 4
                    col = i % 4
                    img = real_data[i].cpu().squeeze().numpy()
                    # Denormalize from [-1, 1] to [0, 1]
                    img = (img + 1) / 2
                    axes[row, col].imshow(img, cmap='gray')
                    axes[row, col].axis('off')
                    axes[row, col].set_title(f'Real {i+1}')
            
            plt.suptitle(f'Generated vs Real Images - Epoch {epoch}')
            plt.tight_layout()
            plt.savefig(f'output/samples/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        self.generator.train()
        print(f"Samples saved for epoch {epoch}")
    
    def save_quick_sample(self, epoch, batch_idx, fixed_noise):
        """
        Save a quick single sample image after each generator run
        
        Args:
            epoch (int): Current epoch
            batch_idx (int): Current batch index
            fixed_noise (torch.Tensor): Fixed noise for consistent generation
        """
        self.generator.eval()
        with torch.no_grad():
            # Generate a single sample using fixed noise
            fake_sample = self.generator(fixed_noise[:1])  # Just use first noise vector
            
            # Convert to displayable format
            img = fake_sample[0].cpu().squeeze().numpy()
            img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            img = np.clip(img, 0, 1)
            
            # Save the image
            plt.figure(figsize=(4, 4))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(f'E{epoch:02d}_B{batch_idx:03d}')
            plt.tight_layout()
            
            # Create filename with epoch and batch info
            filename = f'output/batch_samples/epoch_{epoch:02d}_batch_{batch_idx:03d}.png'
            plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        
        self.generator.train()
    
    def save_models(self, epoch, final=False):
        """Save model checkpoints"""
        if final:
            torch.save(self.generator.state_dict(), 'output/models/generator_final.pth')
            torch.save(self.discriminator.state_dict(), 'output/models/discriminator_final.pth')
            print("Final models saved!")
        else:
            torch.save(self.generator.state_dict(), f'output/models/generator_epoch_{epoch:03d}.pth')
            torch.save(self.discriminator.state_dict(), f'output/models/discriminator_epoch_{epoch:03d}.pth')
    
    def plot_losses(self, epoch):
        """Plot training losses and discriminator accuracies"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        ax1.plot(self.G_losses, label='Generator Loss', color='blue')
        ax1.plot(self.D_losses, label='Discriminator Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('GAN Training Losses')
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
        
        # Plot generator loss trend (smoothed)
        if len(self.G_losses) > 5:
            # Simple moving average for trend
            window = min(5, len(self.G_losses))
            g_loss_smooth = np.convolve(self.G_losses, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(self.G_losses)), g_loss_smooth, label='G Loss (smoothed)', color='blue')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Smoothed Loss')
            ax3.set_title('Generator Loss Trend')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot loss ratio (G_loss / D_loss)
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
        plt.savefig(f'output/samples/training_metrics_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_samples(self, num_samples=1, save_path=None):
        """
        Generate samples using the trained generator
        
        Args:
            num_samples (int): Number of samples to generate
            save_path (str): Path to save the samples
            
        Returns:
            numpy.ndarray: Generated samples
        """
        self.generator.eval()
        with torch.no_grad():
            fake_data = self.generate_fake_data(num_samples)
            # Denormalize from [-1, 1] to [0, 1]
            samples = ((fake_data.cpu().numpy() + 1) / 2)
            
            if save_path:
                np.save(save_path, samples)
                print(f"Samples saved to {save_path}")
        
        self.generator.train()
        return samples


def main():
    """Main training function"""
    # Configuration
    config = {
        'data_dir': '../data',  # Directory containing TIF files (one level up from models folder)
        'latent_dim': 100,
        'image_size': 256,
        'lr': 0.0001,  # Reduced learning rate for more stable training
        'num_epochs': 100,  # Increased epochs for better convergence
        'batch_size': 4,  # Small batch size for 256x256 images
        'save_interval': 5,
        'quick_sample_freq': 5  # Save quick sample every 5 batches
    }
    
    print("GAN Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check if data directory exists
    if not os.path.exists(config['data_dir']):
        print(f"\nError: Data directory '{config['data_dir']}' not found!")
        print("Please create a 'data' directory and place your TIF files there.")
        print("You can organize them in subdirectories if needed.")
        return
    
    try:
        # Initialize and train GAN
        gan = GANTrainer(
            data_dir=config['data_dir'],
            latent_dim=config['latent_dim'],
            image_size=config['image_size'],
            lr=config['lr']
        )
        
        # Train the GAN
        gan.train(
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            save_interval=config['save_interval'],
            quick_sample_freq=config['quick_sample_freq']
        )
        
        # Generate some final samples
        print("\nGenerating final samples...")
        samples = gan.generate_samples(num_samples=16, save_path='output/final_samples.npy')
        
        print("Training completed successfully!")
        print("Check the 'output/' directory for:")
        print("  - Generated samples vs real image comparisons")
        print("  - Saved model checkpoints")
        print("  - Training loss plots")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Make sure you have TIF files in the data directory and PyTorch installed.")


if __name__ == "__main__":
    main()
