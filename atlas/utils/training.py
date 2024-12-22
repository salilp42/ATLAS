#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training utilities for ATLAS medical diffusion model.
"""

import os
from typing import Optional, Dict

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class DiffusionTrainer:
    """Trainer class for the diffusion model."""
    
    def __init__(self, model, diffusion, config):
        self.model = model
        self.diffusion = diffusion
        self.config = config
        self.device = config.DEVICE
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LR,
            betas=config.BETAS
        )
        self.ema = EMA(model, decay=config.EMA_DECAY)
        self.scaler = GradScaler(enabled=config.USE_AMP)
        
        if WANDB_AVAILABLE:
            wandb.init(project=config.WANDB_PROJECT, name=config.WANDB_RUN_NAME)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Get data
        real_img = batch["image"].to(self.device)
        
        # Sample timestep
        t = torch.randint(0, self.config.TIMESTEPS, (self.config.BATCH_SIZE,),
                         device=self.device).long()
        
        # Add noise
        noise = torch.randn_like(real_img)
        x_noisy = self.diffusion.forward_diffusion_sample(real_img, t, noise)

        # Model prediction
        with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
            noise_pred = self.model(x_noisy, t, batch)
            loss = F.mse_loss(noise_pred, noise)

        # Backward pass
        self.scaler.scale(loss).backward()
        
        if self.config.GRAD_CLIP:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.GRAD_CLIP
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.update()

        return loss.item()

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ema_shadow': self.ema.shadow,
            'config': self.config.to_dict()
        }
        path = os.path.join(self.config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ema.shadow = checkpoint['ema_shadow']
        return checkpoint['epoch']

    def generate_samples(self, n_samples: int = 4):
        """Generate samples using the model."""
        self.model.eval()
        self.ema.apply_shadow()

        # Start from noise
        x = torch.randn(n_samples, self.config.IN_CHANNELS,
                       self.config.IMG_SIZE, self.config.IMG_SIZE,
                       device=self.device)

        # Dummy conditioning
        cond_dict = {
            "text": torch.zeros(n_samples, self.config.TEXT_EMB_DIM, device=self.device),
            "ehr": torch.zeros(n_samples, self.config.EHR_EMB_DIM, device=self.device),
            "mask": torch.zeros(n_samples, 1, self.config.IMG_SIZE,
                              self.config.IMG_SIZE, device=self.device)
        }

        # Denoise
        for i in reversed(range(self.config.TIMESTEPS)):
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            with torch.no_grad():
                x = self.diffusion.sample_timestep(x, t, self.model, cond_dict)

        self.ema.restore()
        return x.clamp(-1, 1)

def train(config):
    """Main training loop."""
    from atlas.data.dataset import get_dataloader
    from atlas.models.unet import AtlasDiffusionModel
    from atlas.utils.diffusion import DiffusionHelper
    
    # Setup
    train_loader = get_dataloader(config, is_training=True)
    model = AtlasDiffusionModel(config).to(config.DEVICE)
    diffusion = DiffusionHelper(config)
    trainer = DiffusionTrainer(model, diffusion, config)

    # Training loop
    for epoch in range(config.EPOCHS):
        total_loss = 0
        for batch in train_loader:
            loss = trainer.train_step(batch)
            total_loss += loss

        # Log epoch metrics
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] - Loss: {avg_loss:.4f}")
        
        if WANDB_AVAILABLE:
            wandb.log({"epoch": epoch, "loss": avg_loss})

        # Generate and save samples
        if (epoch + 1) % 5 == 0:
            samples = trainer.generate_samples()
            grid = vutils.make_grid((samples + 1)*0.5, nrow=2)
            plt.figure(figsize=(8,8))
            plt.imshow(grid.permute(1,2,0).cpu().numpy())
            plt.axis('off')
            plt.savefig(os.path.join(config.OUTPUT_DIR, f'samples_epoch_{epoch+1}.png'))
            plt.close()

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch)
