#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for ATLAS medical diffusion model.
"""

import os
import torch

class Config:
    """Configuration class for ATLAS model and training."""
    
    def __init__(self, **kwargs):
        # Data paths
        self.DATA_ROOT = kwargs.get("DATA_ROOT", "data/multimodal")
        self.OUTPUT_DIR = kwargs.get("OUTPUT_DIR", "outputs")
        self.CHECKPOINT_DIR = kwargs.get("CHECKPOINT_DIR", "checkpoints")
        
        # Create directories
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

        # Model architecture
        self.IMG_SIZE = kwargs.get("IMG_SIZE", 256)
        self.IN_CHANNELS = kwargs.get("IN_CHANNELS", 1)
        self.BASE_CHANNELS = kwargs.get("BASE_CHANNELS", 64)
        self.TIME_EMB_DIM = kwargs.get("TIME_EMB_DIM", 256)
        self.TEXT_EMB_DIM = kwargs.get("TEXT_EMB_DIM", 768)
        self.EHR_EMB_DIM = kwargs.get("EHR_EMB_DIM", 128)

        # Training parameters
        self.TIMESTEPS = kwargs.get("TIMESTEPS", 1000)
        self.BATCH_SIZE = kwargs.get("BATCH_SIZE", 4)
        self.EPOCHS = kwargs.get("EPOCHS", 100)
        self.LR = kwargs.get("LR", 2e-4)
        self.BETAS = kwargs.get("BETAS", (0.9, 0.999))
        self.GRAD_CLIP = kwargs.get("GRAD_CLIP", 1.0)
        self.USE_AMP = kwargs.get("USE_AMP", True)
        self.EMA_DECAY = kwargs.get("EMA_DECAY", 0.9999)

        # Model features
        self.CFPG_ENABLED = kwargs.get("CFPG_ENABLED", True)
        self.USE_CAUSAL_ROUTING = kwargs.get("USE_CAUSAL_ROUTING", True)
        self.CAUSAL_GRAPH_PATH = kwargs.get("CAUSAL_GRAPH_PATH", "data/causal_graph.json")

        # Privacy settings
        self.DP_ENABLED = kwargs.get("DP_ENABLED", False)
        self.DP_NOISE_MULTIPLIER = kwargs.get("DP_NOISE_MULTIPLIER", 1.0)

        # Logging
        self.WANDB_PROJECT = kwargs.get("WANDB_PROJECT", "ATLAS")
        self.WANDB_RUN_NAME = kwargs.get("WANDB_RUN_NAME", "default_run")
        self.RUN_ABLATION = kwargs.get("RUN_ABLATION", False)

        # Device
        self.DEVICE = kwargs.get("DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    @classmethod
    def from_dict(cls, config_dict):
        """Create a Config instance from a dictionary."""
        return cls(**config_dict)

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
