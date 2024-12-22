#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model components for the ATLAS medical diffusion model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Positional Embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb

class TimeEmbeddingMLP(nn.Module):
    """MLP for processing time embeddings."""
    def __init__(self, time_emb_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

class AdaGN(nn.Module):
    """Adaptive Group Normalization."""
    def __init__(self, num_channels, emb_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=8, num_channels=num_channels, eps=1e-6)
        self.affine = nn.Linear(emb_channels, num_channels * 2)

    def forward(self, x, emb):
        x_norm = self.groupnorm(x)
        h = self.affine(emb)
        gamma, beta = torch.chunk(h, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x_norm * (1 + gamma) + beta

class CrossAttentionBlock(nn.Module):
    """Cross-attention mechanism for multi-modal fusion."""
    def __init__(self, dim, context_dim, num_heads=4):
        super().__init__()
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Linear(context_dim, dim)
        self.value = nn.Linear(context_dim, dim)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.num_heads = num_heads

    def forward(self, x, context):
        B, C, H, W = x.shape
        q = self.query(x).reshape(B, C, H*W)   
        q = q.permute(0, 2, 1)

        if context.dim() == 2:
            context = context.unsqueeze(1)
        _, L, ctx_dim = context.shape
        k = self.key(context).reshape(B, L, C)
        v = self.value(context).reshape(B, L, C)

        attn_scores = torch.bmm(q, k.permute(0, 2, 1)) / math.sqrt(C)
        attn_probs = attn_scores.softmax(dim=-1)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)
        return self.proj_out(attn_output) + x

class ClinicalFeaturePreservationGate(nn.Module):
    """Gate mechanism for preserving clinical features."""
    def __init__(self, in_channels, feature_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.projector = nn.Conv2d(in_channels, feature_dim, kernel_size=1)

    def forward(self, x, clinical_feature_bank=None):
        feats = self.projector(x)
        x_preserved = x * self.gamma + self.beta
        return x_preserved

class AnatomicalPriorNetwork(nn.Module):
    """Network for incorporating anatomical priors."""
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x)

class CausalRouting(nn.Module):
    """Causal routing mechanism based on learned or predefined graphs."""
    def __init__(self, config, graph=None):
        super().__init__()
        self.config = config
        self.graph = graph
        self.transform = nn.Linear(config.BASE_CHANNELS * 4, config.BASE_CHANNELS * 4)

    def forward(self, feats):
        B, C, H, W = feats.shape
        reshaped = feats.view(B, -1)
        routed = self.transform(reshaped)
        return routed.view(B, C, H, W)

class ResidualBlock(nn.Module):
    """Residual block with attention and CFPG support."""
    def __init__(self, in_channels, out_channels, emb_channels, use_cfpg=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_cfpg = use_cfpg

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.adagn1 = AdaGN(out_channels, emb_channels)
        self.adagn2 = AdaGN(out_channels, emb_channels)
        self.cross_attn = CrossAttentionBlock(dim=out_channels, context_dim=emb_channels)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.cfpg = ClinicalFeaturePreservationGate(out_channels) if use_cfpg else None

    def forward(self, x, emb, context=None):
        h = self.conv1(x)
        h = self.adagn1(h, emb)
        h = self.act(h)

        if context is not None:
            h = self.cross_attn(h, context)

        h = self.conv2(h)
        h = self.adagn2(h, emb)
        h_res = self.act(h + self.skip(x))

        if self.cfpg is not None:
            h_res = self.cfpg(h_res)

        return h_res
