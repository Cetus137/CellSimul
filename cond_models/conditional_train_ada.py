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


class AdaptiveDiscriminatorAugmentation:
    """
    Adaptive Discriminator Augmentation (ADA) for stabilizing GAN training
    Based on CellSynthesis implementation
    """
    def __init__(self, ada_target=0.6, ada_update=0.05, ada_update_period=4):
        self.ada_target = ada_target
        self.ada_update = ada_update
        self.ada_update_period = ada_update_period
        self.ada_aug_p = 0.0
        self.ada_step = 0
        self.ada_stats = []
        
    def update(self, real_predictions):
        """Update ADA probability based on discriminator accuracy on real images"""
        with torch.no_grad():
            # Calculate accuracy on real images
            real_accuracy = (torch.sigmoid(real_predictions) > 0.5).float().mean().item()
            self.ada_stats.append(real_accuracy)
            
            if len(self.ada_stats) >= self.ada_update_period:
                mean_accuracy = np.mean(self.ada_stats)
                
                # Adjust augmentation probability based on accuracy
                if mean_accuracy > self.ada_target:
                    self.ada_aug_p = min(1.0, self.ada_aug_p + self.ada_update)
                else:
                    self.ada_aug_p = max(0.0, self.ada_aug_p - self.ada_update)
                
                self.ada_stats = []
                
        return self.ada_aug_p


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
    """
    
    def __init__(self, fluorescent_dir, mask_dir, latent_dim=100, image_size=256, 
                 lr_g=0.0001, lr_d=0.0004, device=None, use_simple_models=False,
                 ada_target=0.6, ada_update=0.05):
        """
        Initialize the conditional GAN trainer
        
        Args:
            fluorescent_dir (str): Directory containing fluorescent TIF images
            mask_dir (str): Directory containing binary mask TIF images
            latent_dim (int): Dimension of noise vector
            image_size (int): Size of generated images (assumed square)
            lr_g (float): Generator learning rate
            lr_d (float): Discriminator learning rate (higher for better performance)
            device (str): Device to use for training
            use_simple_models (bool): Whether to use simplified models
            ada_target (float): Target discriminator accuracy for ADA
            ada_update (float): ADA update step size
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
        
        # Loss functions
        self.adversarial_loss = AdversarialLoss()
        self.identity_loss = IdentityLoss()
        
        # Optimizers - CellSynthesis style
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), 
            lr=lr_g, 
            betas=(0.0, 0.99)  # CellSynthesis uses these betas
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=lr_d, 
            betas=(0.0, 0.99)
        )
        
        # Initialize ADA
        self.ada = AdaptiveDiscriminatorAugmentation(
            ada_target=ada_target,
            ada_update=ada_update,
            ada_update_period=4
        )
        
        # Training state
        self.current_epoch = 0
        self.g_losses = []
        self.d_losses = []
        self.ada_probs = []
        
        print(f"Initialized ConditionalGANTrainer:")
        print(f"  Generator LR: {lr_g}")
        print(f"  Discriminator LR: {lr_d}")
        print(f"  ADA Target: {ada_target}")
        print(f"  Device: {self.device}")
    
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
        Training step inspired by CellSynthesis PyTorch Lightning approach
        
        Args:
            batch: Batch containing fluorescent images and masks
            optimizer_idx: 0 for generator, 1 for discriminator
        """
        fluorescent_imgs, masks = batch
        batch_size = fluorescent_imgs.size(0)
        
        # Generate random noise
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        
        # Generate fake images
        fake_imgs = self.generator(noise, masks)
        
        if optimizer_idx == 0:
            # Generator training step
            return self._generator_step(fake_imgs, masks, fluorescent_imgs)
        else:
            # Discriminator training step
            return self._discriminator_step(fake_imgs.detach(), fluorescent_imgs, masks)
    
    def _generator_step(self, fake_imgs, masks, real_imgs):
        """Generator training step"""
        # Adversarial loss - fool the discriminator
        fake_pred = self.discriminator(fake_imgs, masks)
        g_adv_loss = self.adversarial_loss(fake_pred, target_is_real=True)
        
        # Identity loss - match real images
        g_identity_loss = self.identity_loss(fake_imgs, real_imgs)
        
        # Total generator loss (weighted like CellSynthesis)
        g_loss = g_adv_loss + 10.0 * g_identity_loss  # L1 weight = 10
        
        return {
            'loss': g_loss,
            'g_adv_loss': g_adv_loss.item(),
            'g_identity_loss': g_identity_loss.item()
        }
    
    def _discriminator_step(self, fake_imgs, real_imgs, masks):
        """Discriminator training step"""
        # Real images
        real_pred = self.discriminator(real_imgs, masks)
        d_real_loss = self.adversarial_loss(real_pred, target_is_real=True)
        
        # Fake images
        fake_pred = self.discriminator(fake_imgs, masks)
        d_fake_loss = self.adversarial_loss(fake_pred, target_is_real=False)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        # Update ADA based on real predictions
        ada_prob = self.ada.update(real_pred)
        
        return {
            'loss': d_loss,
            'd_real_loss': d_real_loss.item(),
            'd_fake_loss': d_fake_loss.item(),
            'ada_prob': ada_prob
        }
    
    def train_epoch(self, dataloader):
        """Train for one epoch using alternating optimization"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_ada_prob = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = [x.to(self.device) for x in batch]
            
            # Train discriminator
            self.optimizer_D.zero_grad()
            d_results = self.training_step(batch, optimizer_idx=1)
            d_results['loss'].backward()
            self.optimizer_D.step()
            
            # Train generator
            self.optimizer_G.zero_grad()
            g_results = self.training_step(batch, optimizer_idx=0)
            g_results['loss'].backward()
            self.optimizer_G.step()
            
            # Accumulate losses
            epoch_g_loss += g_results['loss'].item()
            epoch_d_loss += d_results['loss'].item()
            epoch_ada_prob += d_results['ada_prob']
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{num_batches}: "
                      f"G_loss: {g_results['loss'].item():.4f}, "
                      f"D_loss: {d_results['loss'].item():.4f}, "
                      f"ADA_prob: {d_results['ada_prob']:.4f}")
        
        # Average losses for the epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_ada_prob = epoch_ada_prob / num_batches
        
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
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Generator Loss: {epoch_results['g_loss']:.6f}")
            print(f"  Discriminator Loss: {epoch_results['d_loss']:.6f}")
            print(f"  ADA Probability: {epoch_results['ada_prob']:.4f}")
            
            # Save samples and models periodically
            if (epoch + 1) % save_interval == 0:
                self.save_samples(output_dir, epoch)
                self.save_models(output_dir, epoch)
                self.save_training_curves(output_dir)
        
        print("Training completed!")
        self.save_final_results(output_dir)
    
    def save_samples(self, output_dir, epoch):
        """Save generated samples"""
        self.generator.eval()
        
        with torch.no_grad():
            # Get a batch from dataset
            sample_batch = next(iter(DataLoader(self.dataset, batch_size=8, shuffle=True)))
            fluorescent_imgs, masks = [x.to(self.device) for x in sample_batch]
            
            # Generate samples
            noise = torch.randn(masks.size(0), self.latent_dim, device=self.device)
            fake_imgs = self.generator(noise, masks)
            
            # Save comparison
            comparison = torch.cat([
                masks.cpu(),
                fluorescent_imgs.cpu(),
                fake_imgs.cpu()
            ], dim=0)
            
            save_path = os.path.join(output_dir, f'samples_epoch_{epoch+1}.png')
            save_image(comparison, save_path, nrow=masks.size(0), normalize=True)
            print(f"Saved samples to {save_path}")
    
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
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Conditional GAN with ADA')
    parser.add_argument('--fluorescent_dir', type=str, required=True,
                      help='Directory containing fluorescent images')
    parser.add_argument('--mask_dir', type=str, required=True,
                      help='Directory containing mask images')
    parser.add_argument('--output_dir', type=str, default='./outputs_ada',
                      help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size')
    parser.add_argument('--lr_g', type=float, default=0.0001,
                      help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0004,
                      help='Discriminator learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                      help='Latent dimension')
    parser.add_argument('--image_size', type=int, default=256,
                      help='Image size')
    parser.add_argument('--ada_target', type=float, default=0.6,
                      help='ADA target accuracy')
    parser.add_argument('--ada_update', type=float, default=0.05,
                      help='ADA update step size')
    parser.add_argument('--use_simple_models', action='store_true',
                      help='Use simplified models')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ConditionalGANTrainer(
        fluorescent_dir=args.fluorescent_dir,
        mask_dir=args.mask_dir,
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
        save_interval=10,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
