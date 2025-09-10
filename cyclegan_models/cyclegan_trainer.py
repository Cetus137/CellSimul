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

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from cyclegan_networks import CycleGANGenerator, CycleGANDiscriminator, CycleGANLoss, init_weights
from cond_models.unpaired_conditional_dataloader import UnpairedConditionalImageDataset


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
    CycleGAN trainer for mask-to-fluorescent image translation
    """
    
    def __init__(self, mask_dir, fluorescent_dir, image_size=256, 
                 cycle_loss_weight=10.0, identity_loss_weight=0.5,
                 lr=0.0002, beta1=0.5, device=None, pretrained_checkpoint=None):
        """
        Initialize CycleGAN trainer
        
        Args:
            mask_dir: Directory containing distance masks
            fluorescent_dir: Directory containing fluorescent images
            image_size: Size of input images
            cycle_loss_weight: Weight for cycle consistency loss
            identity_loss_weight: Weight for identity loss
            lr: Learning rate
            beta1: Adam optimizer beta1 parameter
            device: Training device
            pretrained_checkpoint: Path to pretrained checkpoint file (.pth)
        """
        
        self.mask_dir = mask_dir
        self.fluorescent_dir = fluorescent_dir
        self.image_size = image_size
        self.lr = lr
        self.beta1 = beta1
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.G_M2F = CycleGANGenerator(input_channels=1, output_channels=1).to(self.device)  # Mask to Fluorescent
        self.G_F2M = CycleGANGenerator(input_channels=1, output_channels=1).to(self.device)  # Fluorescent to Mask
        self.D_F = CycleGANDiscriminator(input_channels=1).to(self.device)  # Discriminator for fluorescent domain
        self.D_M = CycleGANDiscriminator(input_channels=1).to(self.device)  # Discriminator for mask domain
        
        # Initialize weights
        init_weights(self.G_M2F, init_type='normal', init_gain=0.02)
        init_weights(self.G_F2M, init_type='normal', init_gain=0.02)
        init_weights(self.D_F, init_type='normal', init_gain=0.02)
        init_weights(self.D_M, init_type='normal', init_gain=0.02)
        
        print("Initialized CycleGAN networks:")
        print(f"  G_M2F: Mask → Fluorescent")
        print(f"  G_F2M: Fluorescent → Mask") 
        print(f"  D_F: Fluorescent discriminator")
        print(f"  D_M: Mask discriminator")
        
        # Loss function
        self.criterion = CycleGANLoss(cycle_loss_weight, identity_loss_weight)
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            list(self.G_M2F.parameters()) + list(self.G_F2M.parameters()),
            lr=lr, betas=(beta1, 0.999)
        )
        self.optimizer_D_F = optim.Adam(self.D_F.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D_M = optim.Adam(self.D_M.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
        )
        self.scheduler_D_F = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_F, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
        )
        self.scheduler_D_M = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_M, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
        )
        
        # Mixed precision scalers (for faster training)
        self.scaler_G = GradScaler() if self.device.type == 'cuda' else None
        self.scaler_D_F = GradScaler() if self.device.type == 'cuda' else None
        self.scaler_D_M = GradScaler() if self.device.type == 'cuda' else None
        
        # Image buffers for discriminator stability
        self.fake_F_buffer = ImageBuffer(50)
        self.fake_M_buffer = ImageBuffer(50)
        
        # Load dataset
        self.dataset = UnpairedConditionalImageDataset(
            fluorescent_dir=fluorescent_dir,
            mask_dir=mask_dir,
            image_size=image_size,
            max_images=None  # Use all available images
        )
        
        dataset_info = self.dataset.get_dataset_info()
        print(f"\nDataset loaded:")
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")
        
        # Training metrics
        self.g_losses = []
        self.d_f_losses = []
        self.d_m_losses = []
        self.start_epoch = 0
        
        # Load pretrained checkpoint if provided
        if pretrained_checkpoint is not None:
            self.load_checkpoint(pretrained_checkpoint)
    
    def load_checkpoint(self, checkpoint_path):
        """Load pretrained checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"\nLoading pretrained checkpoint from: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model states
            self.G_M2F.load_state_dict(checkpoint['G_M2F_state_dict'])
            self.G_F2M.load_state_dict(checkpoint['G_F2M_state_dict'])
            self.D_F.load_state_dict(checkpoint['D_F_state_dict'])
            self.D_M.load_state_dict(checkpoint['D_M_state_dict'])
            
            # Load optimizer states
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.optimizer_D_F.load_state_dict(checkpoint['optimizer_D_F_state_dict'])
            self.optimizer_D_M.load_state_dict(checkpoint['optimizer_D_M_state_dict'])
            
            # Load training history
            self.g_losses = checkpoint.get('g_losses', [])
            self.d_f_losses = checkpoint.get('d_f_losses', [])
            self.d_m_losses = checkpoint.get('d_m_losses', [])
            
            # Set starting epoch
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            
            print(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
            print(f"Resuming training from epoch {self.start_epoch}")
            print(f"Loaded training history: {len(self.g_losses)} epochs")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            self.start_epoch = 0
    
    def train_step(self, real_masks, real_fluorescent):
        """Single training step with mixed precision"""
        batch_size = real_masks.size(0)
        
        # Set model modes
        self.G_M2F.train()
        self.G_F2M.train()
        self.D_F.train()
        self.D_M.train()
        
        # ------------------
        # Train Generators
        # ------------------
        self.optimizer_G.zero_grad()
        
        with autocast(enabled=self.scaler_G is not None):
            # Identity loss (optional - can be disabled by setting weight to 0)
            same_F = self.G_M2F(real_fluorescent)  # G_M2F should be identity for fluorescent input
            same_M = self.G_F2M(real_masks)        # G_F2M should be identity for mask input
            
            # Forward cycle: Mask → Fluorescent → reconstructed Mask
            fake_fluorescent = self.G_M2F(real_masks)
            reconstructed_masks = self.G_F2M(fake_fluorescent)
            
            # Backward cycle: Fluorescent → Mask → reconstructed Fluorescent
            fake_masks = self.G_F2M(real_fluorescent)
            reconstructed_fluorescent = self.G_M2F(fake_masks)
            
            # Adversarial losses
            pred_fake_F = self.D_F(fake_fluorescent)
            pred_fake_M = self.D_M(fake_masks)
            
            # Generator losses
            g_m2f_loss = self.criterion.generator_loss(
                pred_fake_F, real_masks, real_fluorescent, fake_fluorescent, fake_masks,
                reconstructed_masks, reconstructed_fluorescent, same_M, same_F
            )
            
            g_f2m_loss = self.criterion.generator_loss(
                pred_fake_M, real_fluorescent, real_masks, fake_masks, fake_fluorescent,
                reconstructed_fluorescent, reconstructed_masks, same_F, same_M
            )
            
            # Total generator loss
            g_total_loss = g_m2f_loss['total_loss'] + g_f2m_loss['total_loss']
        
        if self.scaler_G is not None:
            self.scaler_G.scale(g_total_loss).backward()
            self.scaler_G.step(self.optimizer_G)
            self.scaler_G.update()
        else:
            g_total_loss.backward()
            self.optimizer_G.step()
        
        # ------------------
        # Train Discriminator F (Fluorescent)
        # ------------------
        self.optimizer_D_F.zero_grad()
        
        with autocast(enabled=self.scaler_D_F is not None):
            # Real fluorescent images
            pred_real_F = self.D_F(real_fluorescent)
            
            # Fake fluorescent images from buffer
            fake_fluorescent_buffer = self.fake_F_buffer.push_and_pop(fake_fluorescent)
            pred_fake_F = self.D_F(fake_fluorescent_buffer.detach())
            
            # Discriminator F loss
            d_f_loss = self.criterion.discriminator_loss(pred_real_F, pred_fake_F)
        
        if self.scaler_D_F is not None:
            self.scaler_D_F.scale(d_f_loss).backward()
            self.scaler_D_F.step(self.optimizer_D_F)
            self.scaler_D_F.update()
        else:
            d_f_loss.backward()
            self.optimizer_D_F.step()
        
        # ------------------
        # Train Discriminator M (Mask)
        # ------------------
        self.optimizer_D_M.zero_grad()
        
        with autocast(enabled=self.scaler_D_M is not None):
            # Real mask images
            pred_real_M = self.D_M(real_masks)
            
            # Fake mask images from buffer
            fake_masks_buffer = self.fake_M_buffer.push_and_pop(fake_masks)
            pred_fake_M = self.D_M(fake_masks_buffer.detach())
            
            # Discriminator M loss
            d_m_loss = self.criterion.discriminator_loss(pred_real_M, pred_fake_M)
        
        if self.scaler_D_M is not None:
            self.scaler_D_M.scale(d_m_loss).backward()
            self.scaler_D_M.step(self.optimizer_D_M)
            self.scaler_D_M.update()
        else:
            d_m_loss.backward()
            self.optimizer_D_M.step()
        
        return {
            'g_loss': g_total_loss.item(),
            'g_m2f_adv': g_m2f_loss['adv_loss'],
            'g_f2m_adv': g_f2m_loss['adv_loss'],
            'g_cycle': g_m2f_loss['cycle_loss'] + g_f2m_loss['cycle_loss'],
            'g_identity': g_m2f_loss['identity_loss'] + g_f2m_loss['identity_loss'],
            'd_f_loss': d_f_loss.item(),
            'd_m_loss': d_m_loss.item()
        }
    
    def train(self, num_epochs, batch_size, save_interval=10, output_dir='./cyclegan_outputs', num_workers=4):
        """Main training loop"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dataloader
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        
        print(f"\nStarting CycleGAN training:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Starting from epoch: {self.start_epoch + 1}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {len(dataloader)}")
        print(f"  Save interval: {save_interval}")
        print(f"  Output directory: {output_dir}")
        
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            epoch_g_loss = 0
            epoch_d_f_loss = 0
            epoch_d_m_loss = 0
            
            for batch_idx, (real_fluorescent, real_masks) in enumerate(dataloader):
                real_fluorescent = real_fluorescent.to(self.device)
                real_masks = real_masks.to(self.device)
                
                # Training step
                losses = self.train_step(real_masks, real_fluorescent)
                
                epoch_g_loss += losses['g_loss']
                epoch_d_f_loss += losses['d_f_loss']
                epoch_d_m_loss += losses['d_m_loss']
                
                # Print progress
                if batch_idx % 25 == 0:
                    print(f"Epoch {epoch+1}/{self.start_epoch + num_epochs}, Batch {batch_idx}/{len(dataloader)}: "
                          f"G: {losses['g_loss']:.4f} "
                          f"(adv: {losses['g_m2f_adv']:.3f}+{losses['g_f2m_adv']:.3f}, "
                          f"cycle: {losses['g_cycle']:.3f}, id: {losses['g_identity']:.3f}), "
                          f"D_F: {losses['d_f_loss']:.4f}, D_M: {losses['d_m_loss']:.4f}")
            
            # Average losses for the epoch
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_f_loss = epoch_d_f_loss / len(dataloader)
            avg_d_m_loss = epoch_d_m_loss / len(dataloader)
            
            self.g_losses.append(avg_g_loss)
            self.d_f_losses.append(avg_d_f_loss)
            self.d_m_losses.append(avg_d_m_loss)
            
            # Update learning rates
            self.scheduler_G.step()
            self.scheduler_D_F.step()
            self.scheduler_D_M.step()
            
            print(f"\nEpoch {epoch+1}/{num_epochs} completed:")
            print(f"  Generator Loss: {avg_g_loss:.6f}")
            print(f"  Discriminator F Loss: {avg_d_f_loss:.6f}")
            print(f"  Discriminator M Loss: {avg_d_m_loss:.6f}")
            
            # Save samples and model
            if (epoch + 1) % save_interval == 0:
                self.save_samples(epoch, output_dir)
                self.save_model(epoch, output_dir)
                self.plot_losses(output_dir)
        
        print("Training completed!")
        self.save_final_results(output_dir)
    
    def save_samples(self, epoch, output_dir):
        """Save generated samples as individual grayscale TIFF files"""
        self.G_M2F.eval()
        self.G_F2M.eval()
        
        with torch.no_grad():
            # Get test samples
            test_dataloader = DataLoader(self.dataset, batch_size=8, shuffle=True)
            real_fluorescent, real_masks = next(iter(test_dataloader))
            real_fluorescent = real_fluorescent.to(self.device)
            real_masks = real_masks.to(self.device)
            
            # Generate translations
            fake_fluorescent = self.G_M2F(real_masks)
            fake_masks = self.G_F2M(real_fluorescent)
            
            # Cycle reconstructions
            reconstructed_masks = self.G_F2M(fake_fluorescent)
            reconstructed_fluorescent = self.G_M2F(fake_masks)
            
            # Normalize tensor to [0, 255] range for TIFF saving
            def normalize_to_uint8(tensor):
                """Normalize tensor to [0, 255] range as uint8"""
                # Clamp to handle any extreme values
                tensor = torch.clamp(tensor, -1, 1)
                # Convert from [-1, 1] to [0, 255]
                tensor = ((tensor + 1.0) / 2.0 * 255.0).round()
                return tensor.cpu().numpy().astype(np.uint8)
            
            # Create epoch directory
            epoch_dir = os.path.join(output_dir, f'epoch_{epoch+1:03d}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Apply normalization and save each image type
            real_masks_np = normalize_to_uint8(real_masks)
            fake_fluorescent_np = normalize_to_uint8(fake_fluorescent)
            reconstructed_masks_np = normalize_to_uint8(reconstructed_masks)
            real_fluorescent_np = normalize_to_uint8(real_fluorescent)
            fake_masks_np = normalize_to_uint8(fake_masks)
            reconstructed_fluorescent_np = normalize_to_uint8(reconstructed_fluorescent)
            
            # Save individual TIFF files for each sample in the batch
            for i in range(real_masks.size(0)):
                sample_prefix = f'sample_{i:02d}'
                
                # Save real masks
                img = Image.fromarray(real_masks_np[i, 0], mode='L')  # [0] for channel dimension
                img.save(os.path.join(epoch_dir, f'{sample_prefix}_real_mask.tif'))
                
                # Save generated fluorescent (M→F)
                img = Image.fromarray(fake_fluorescent_np[i, 0], mode='L')
                img.save(os.path.join(epoch_dir, f'{sample_prefix}_generated_fluorescent_M2F.tif'))
                
                # Save reconstructed masks (M→F→M)
                img = Image.fromarray(reconstructed_masks_np[i, 0], mode='L')
                img.save(os.path.join(epoch_dir, f'{sample_prefix}_reconstructed_mask_M2F2M.tif'))
                
                # Save real fluorescent
                img = Image.fromarray(real_fluorescent_np[i, 0], mode='L')
                img.save(os.path.join(epoch_dir, f'{sample_prefix}_real_fluorescent.tif'))
                
                # Save generated masks (F→M)
                img = Image.fromarray(fake_masks_np[i, 0], mode='L')
                img.save(os.path.join(epoch_dir, f'{sample_prefix}_generated_mask_F2M.tif'))
                
                # Save reconstructed fluorescent (F→M→F)
                img = Image.fromarray(reconstructed_fluorescent_np[i, 0], mode='L')
                img.save(os.path.join(epoch_dir, f'{sample_prefix}_reconstructed_fluorescent_F2M2F.tif'))
            
            print(f"Saved CycleGAN samples to {epoch_dir}/")
            print(f"  Saved {real_masks.size(0)} samples with 6 image types each")
            print(f"  Image types: real_mask, generated_fluorescent_M2F, reconstructed_mask_M2F2M,")
            print(f"               real_fluorescent, generated_mask_F2M, reconstructed_fluorescent_F2M2F")
            print(f"  Format: Grayscale TIFF, 8-bit, range [0-255]")
    
    def save_model(self, epoch, output_dir):
        """Save model checkpoints"""
        checkpoint = {
            'epoch': epoch,
            'G_M2F_state_dict': self.G_M2F.state_dict(),
            'G_F2M_state_dict': self.G_F2M.state_dict(),
            'D_F_state_dict': self.D_F.state_dict(),
            'D_M_state_dict': self.D_M.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_F_state_dict': self.optimizer_D_F.state_dict(),
            'optimizer_D_M_state_dict': self.optimizer_D_M.state_dict(),
            'g_losses': self.g_losses,
            'd_f_losses': self.d_f_losses,
            'd_m_losses': self.d_m_losses
        }
        
        save_path = os.path.join(output_dir, f'cyclegan_checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, save_path)
        print(f"Saved model checkpoint to {save_path}")
    
    def plot_losses(self, output_dir):
        """Plot training losses"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.g_losses, label='Generator')
        plt.title('Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.d_f_losses, label='Discriminator F', color='orange')
        plt.title('Discriminator F Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(self.d_m_losses, label='Discriminator M', color='green')
        plt.title('Discriminator M Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'cyclegan_training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved training curves to {save_path}")
    
    def save_final_results(self, output_dir):
        """Save final training results"""
        results = {
            'final_g_loss': self.g_losses[-1] if self.g_losses else None,
            'final_d_f_loss': self.d_f_losses[-1] if self.d_f_losses else None,
            'final_d_m_loss': self.d_m_losses[-1] if self.d_m_losses else None,
            'total_epochs': len(self.g_losses),
            'lr': self.lr,
            'device': str(self.device),
            'mask_dir': self.mask_dir,
            'fluorescent_dir': self.fluorescent_dir
        }
        
        with open(os.path.join(output_dir, 'cyclegan_training_summary.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nFinal CycleGAN Results:")
        print(f"  Generator Loss: {results['final_g_loss']:.6f}")
        print(f"  Discriminator F Loss: {results['final_d_f_loss']:.6f}")
        print(f"  Discriminator M Loss: {results['final_d_m_loss']:.6f}")
        print(f"  CycleGAN training completed successfully!")


def main():
    """Main training function for CycleGAN"""
    parser = argparse.ArgumentParser(description='Train CycleGAN for Mask-to-Fluorescent Translation')
    parser.add_argument('--mask_dir', type=str, required=True,
                      help='Directory containing distance masks')
    parser.add_argument('--fluorescent_dir', type=str, required=True,
                      help='Directory containing fluorescent images')
    parser.add_argument('--output_dir', type=str, default='./cyclegan_outputs',
                      help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                      help='Learning rate')
    parser.add_argument('--cycle_loss_weight', type=float, default=10.0,
                      help='Weight for cycle consistency loss')
    parser.add_argument('--identity_loss_weight', type=float, default=0.5,
                      help='Weight for identity loss')
    parser.add_argument('--image_size', type=int, default=256,
                      help='Image size')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--save_interval', type=int, default=1,
                      help='Interval for saving checkpoints and samples')
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                      help='Path to pretrained checkpoint file (.pth) to resume training from')
    parser.add_argument('--fast_mode', action='store_true',
                      help='Enable fast mode: larger batch size, higher learning rate, reduced residual blocks')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Apply fast mode optimizations
    if args.fast_mode:
        print("Fast mode enabled - using optimized settings:")
        args.batch_size = max(args.batch_size, 8)  # Increase batch size
        args.lr = min(args.lr * 2, 0.0004)  # Increase learning rate
        args.save_interval = max(args.save_interval, 5)  # Save less frequently
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Save interval: {args.save_interval}")
    
    # Create CycleGAN trainer
    trainer = CycleGANTrainer(
        mask_dir=args.mask_dir,
        fluorescent_dir=args.fluorescent_dir,
        image_size=args.image_size,
        cycle_loss_weight=args.cycle_loss_weight,
        identity_loss_weight=args.identity_loss_weight,
        lr=args.lr,
        device=args.device,
        pretrained_checkpoint=args.pretrained_checkpoint
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
