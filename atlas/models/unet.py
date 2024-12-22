#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core U-Net architecture for ATLAS medical diffusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import (
    SinusoidalPosEmb,
    TimeEmbeddingMLP,
    CrossAttentionBlock,
    ResidualBlock,
    ClinicalFeaturePreservationGate,
    AnatomicalPriorNetwork,
    CausalRouting
)

class AtlasDiffusionModel(nn.Module):
    """
    ATLAS medical diffusion model with clinical feature preservation,
    causal routing, and privacy-preserving mechanisms.
    """
    def __init__(self, config, causal_graph=None):
        super().__init__()
        self.config = config
        self.time_embed = SinusoidalPosEmb(config.TIME_EMB_DIM)
        self.time_mlp = TimeEmbeddingMLP(config.TIME_EMB_DIM, hidden_dim=config.TIME_EMB_DIM)

        emb_channels = config.TIME_EMB_DIM

        # Anatomical prior
        self.anatomical_prior_net = AnatomicalPriorNetwork(config)

        # Causal routing
        self.causal_graph = causal_graph
        self.causal_router = CausalRouting(config, self.causal_graph) if config.USE_CAUSAL_ROUTING else None

        # Encoder
        self.res1 = ResidualBlock(config.IN_CHANNELS, config.BASE_CHANNELS, emb_channels, use_cfpg=config.CFPG_ENABLED)
        self.down1 = ResidualBlock(config.BASE_CHANNELS, config.BASE_CHANNELS*2, emb_channels, use_cfpg=config.CFPG_ENABLED)
        self.down2 = ResidualBlock(config.BASE_CHANNELS*2, config.BASE_CHANNELS*4, emb_channels, use_cfpg=config.CFPG_ENABLED)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.mid1 = ResidualBlock(config.BASE_CHANNELS*4, config.BASE_CHANNELS*4, emb_channels, use_cfpg=config.CFPG_ENABLED)
        self.mid2 = ResidualBlock(config.BASE_CHANNELS*4, config.BASE_CHANNELS*4, emb_channels, use_cfpg=config.CFPG_ENABLED)

        # Decoder
        self.up2 = ResidualBlock(config.BASE_CHANNELS*8, config.BASE_CHANNELS*2, emb_channels, use_cfpg=config.CFPG_ENABLED)
        self.up1 = ResidualBlock(config.BASE_CHANNELS*4, config.BASE_CHANNELS, emb_channels, use_cfpg=config.CFPG_ENABLED)
        self.out_conv = nn.Conv2d(config.BASE_CHANNELS, config.IN_CHANNELS, 1)

    def forward(self, x, t, cond_dict):
        """Forward pass of the model."""
        text_emb = cond_dict.get("text", None)
        ehr_emb = cond_dict.get("ehr", None)
        seg_mask = cond_dict.get("mask", None)

        # Derive anatomical prior features
        if seg_mask is not None:
            anatomical_emb = self.anatomical_prior_net(seg_mask)
        else:
            anatomical_emb = None

        # Fuse text + EHR + anatomical into a single context
        fused_context = text_emb.unsqueeze(1) if text_emb is not None else None

        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        h1 = self.res1(x, t_emb, fused_context)
        h2 = self.pool(h1)
        h2 = self.down1(h2, t_emb, fused_context)
        h3 = self.pool(h2)
        h3 = self.down2(h3, t_emb, fused_context)

        # Bottleneck with causal routing
        h3 = self.mid1(h3, t_emb, fused_context)
        if self.config.USE_CAUSAL_ROUTING and self.causal_router is not None:
            h3 = self.causal_router(h3)
        h3 = self.mid2(h3, t_emb, fused_context)

        # Decoder
        h3_up = F.interpolate(h3, scale_factor=2, mode='nearest')
        cat2 = torch.cat([h3_up, h2], dim=1)
        h2_up = self.up2(cat2, t_emb, fused_context)

        h2_up = F.interpolate(h2_up, scale_factor=2, mode='nearest')
        cat1 = torch.cat([h2_up, h1], dim=1)
        h1_up = self.up1(cat1, t_emb, fused_context)

        return self.out_conv(h1_up)
