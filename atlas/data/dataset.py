#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset handling for ATLAS medical diffusion model.
"""

import os
from glob import glob
from typing import Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

class SemanticMedicalTokenizer:
    """Text tokenizer for medical descriptions."""
    
    def __init__(self, vocab_size=30522, embed_dim=768):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def encode(self, text: str) -> torch.Tensor:
        """Convert text to embeddings."""
        # Placeholder: return random embedding
        # In practice, use a proper medical text encoder
        return torch.randn(self.embed_dim)

class MultiModalMedicalDataset(Dataset):
    """
    Dataset for multi-modal medical data including:
    - Medical images (X-ray, CT, MRI)
    - Clinical text
    - EHR numeric data
    - Segmentation masks
    """
    
    def __init__(self, config, transform=None, tokenizer=None):
        super().__init__()
        self.config = config
        self.transform = transform or self._default_transform()
        self.tokenizer = tokenizer or SemanticMedicalTokenizer(embed_dim=config.TEXT_EMB_DIM)

        # Load image paths
        self.image_paths = glob(os.path.join(config.DATA_ROOT, "*.png"))  # Adjust pattern as needed
        
        # In practice, also load:
        # - Clinical text data
        # - EHR data
        # - Segmentation masks

    def _default_transform(self):
        """Default image transformations."""
        return T.Compose([
            T.Resize((self.config.IMG_SIZE, self.config.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        image = self.transform(image)

        # Load/generate other modalities
        # In practice, load these from actual data sources
        dummy_text = "Patient shows mild infiltration in left lung."
        text_emb = self.tokenizer.encode(dummy_text)
        ehr_emb = torch.zeros(self.config.EHR_EMB_DIM)
        seg_mask = torch.zeros(1, self.config.IMG_SIZE, self.config.IMG_SIZE)

        return {
            "image": image,
            "text": text_emb,
            "ehr": ehr_emb,
            "mask": seg_mask
        }

def get_dataloader(config, is_training=True):
    """Create data loader with appropriate settings."""
    dataset = MultiModalMedicalDataset(
        config=config,
        transform=None,  # Use default transform
        tokenizer=None   # Use default tokenizer
    )
    
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=is_training,
        num_workers=4,
        pin_memory=True,
        drop_last=is_training
    )
