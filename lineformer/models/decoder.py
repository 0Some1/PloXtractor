"""
Line Query Decoder
Transformer-based decoder that uses learnable line queries to extract line instances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class LineQueryDecoder(nn.Module):
    """
    Transformer decoder with learnable line queries.
    Each query represents one line instance.
    """

    def __init__(
            self,
            d_model: int = 256,
            num_queries: int = 20,
            num_heads: int = 8,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
            max_points: int = 50,
    ):
        """
        Args:
            d_model: Hidden dimension
            num_queries: Number of line queries (max lines per image)
            num_heads: Number of attention heads
            num_decoder_layers: Number of decoder layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            max_points: Maximum points per polyline
        """
        super().__init__()

        self.d_model = d_model
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.max_points = max_points

        # Learnable line queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
            self,
            features: torch.Tensor,
            pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) feature map from backbone
            pos_embed: (B, C, H, W) positional embeddings (optional)

        Returns:
            queries: (B, num_queries, d_model) decoded line query features
        """
        B, C, H, W = features.shape

        # Flatten spatial dimensions: (B, C, H, W) -> (H*W, B, C)
        features_flat = features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)

        # Positional embeddings
        if pos_embed is not None:
            pos_embed_flat = pos_embed.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        else:
            pos_embed_flat = None

        # Initialize queries: (num_queries, B, d_model)
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        # Apply decoder layers
        for layer in self.decoder_layers:
            queries = layer(
                tgt=queries,
                memory=features_flat,
                pos=pos_embed_flat,
            )

        # Final normalization
        queries = self.norm(queries)

        # Permute to (B, num_queries, d_model)
        queries = queries.permute(1, 0, 2)

        return queries


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and cross-attention."""

    def __init__(
            self,
            d_model: int = 256,
            num_heads: int = 8,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention (query-to-query)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=False,
        )

        # Cross-attention (query-to-features)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=False,
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: (num_queries, B, d_model) target queries
            memory: (H*W, B, d_model) encoder features
            pos: (H*W, B, d_model) positional embeddings

        Returns:
            tgt: (num_queries, B, d_model) updated queries
        """
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        if pos is not None:
            memory_with_pos = memory + pos
        else:
            memory_with_pos = memory

        tgt2 = self.cross_attn(tgt, memory_with_pos, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        return tgt


class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for feature maps.
    Uses separate encodings for Y and X coordinates, concatenated together.
    """

    def __init__(self, d_model: int, max_h: int = 256, max_w: int = 256):
        """
        Args:
            d_model: Total dimension (must be divisible by 4)
            max_h: Maximum height to pre-compute
            max_w: Maximum width to pre-compute
        """
        super().__init__()

        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4, got {d_model}")

        self.d_model = d_model
        d_model_half = d_model // 2

        # Create frequency bands
        num_freqs = d_model_half // 2
        div_term = torch.exp(
            torch.arange(0, num_freqs).float() *
            -(math.log(10000.0) / num_freqs)
        )

        # Y-axis positional encoding
        pos_y = torch.arange(max_h).float().unsqueeze(1)  # (max_h, 1)
        pos_y = pos_y * div_term.unsqueeze(0)  # (max_h, num_freqs)

        pe_y = torch.zeros(max_h, d_model_half)
        pe_y[:, 0::2] = torch.sin(pos_y)
        pe_y[:, 1::2] = torch.cos(pos_y)

        # X-axis positional encoding
        pos_x = torch.arange(max_w).float().unsqueeze(1)  # (max_w, 1)
        pos_x = pos_x * div_term.unsqueeze(0)  # (max_w, num_freqs)

        pe_x = torch.zeros(max_w, d_model_half)
        pe_x[:, 0::2] = torch.sin(pos_x)
        pe_x[:, 1::2] = torch.cos(pos_x)

        # Expand to 2D
        # pe_y: (max_h, 1, d_model_half) -> (max_h, max_w, d_model_half)
        # pe_x: (1, max_w, d_model_half) -> (max_h, max_w, d_model_half)
        pe_y = pe_y.unsqueeze(1).expand(max_h, max_w, d_model_half)
        pe_x = pe_x.unsqueeze(0).expand(max_h, max_w, d_model_half)

        # Concatenate: (max_h, max_w, d_model)
        pe = torch.cat([pe_y, pe_x], dim=-1)

        # Reshape to (1, d_model, max_h, max_w) for broadcasting
        pe = pe.permute(2, 0, 1).unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) features

        Returns:
            pos: (B, C, H, W) positional encoding
        """
        B, C, H, W = x.shape

        # Crop to actual size and repeat for batch
        pos_encoding = self.pe[:, :, :H, :W]
        return pos_encoding.expand(B, -1, -1, -1)


class LinePredictionHead(nn.Module):
    """
    Prediction heads for centerline coordinates and line width.
    """

    def __init__(
            self,
            d_model: int = 256,
            max_points: int = 50,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_points = max_points

        # Centerline prediction head (coordinates)
        self.centerline_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, max_points * 2),  # (x, y) for each point
        )

        # Width prediction head (scalar per point)
        self.width_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, max_points),
            nn.Softplus(),  # Ensure positive widths
        )

        # Objectness/validity head (is this a real line?)
        self.objectness_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: (B, num_queries, d_model)

        Returns:
            centerlines: (B, num_queries, max_points, 2) normalized coordinates in [0, 1]
            widths: (B, num_queries, max_points) line widths
            objectness: (B, num_queries) objectness scores
        """
        B, num_queries, _ = queries.shape

        # Predict centerlines
        centerlines_flat = self.centerline_head(queries)  # (B, num_queries, max_points * 2)
        centerlines = centerlines_flat.view(B, num_queries, self.max_points, 2)
        centerlines = torch.sigmoid(centerlines)  # Normalize to [0, 1]

        # Predict widths
        widths = self.width_head(queries)  # (B, num_queries, max_points)

        # Predict objectness
        objectness = self.objectness_head(queries).squeeze(-1)  # (B, num_queries)

        return centerlines, widths, objectness