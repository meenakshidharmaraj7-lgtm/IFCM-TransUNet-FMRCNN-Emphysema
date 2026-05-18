"""
transunet.py

Compact TransUNet-style segmentation network for emphysema CT analysis.

This implementation combines:
- CNN encoder for local texture and boundary features
- Transformer bottleneck for global contextual reasoning
- U-Net decoder with skip connections for fine segmentation

The model is intentionally lightweight enough for reproducibility while still
matching the methodological logic of the IFCM-TransUNet-FMRCNN framework.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Two-layer convolution block used in encoder and decoder.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Encoder downsampling block.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """
    Decoder upsampling block with skip connection.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PatchEmbedding(nn.Module):
    """
    Convert CNN feature maps into transformer tokens.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 1,
        feature_size: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        num_patches = (feature_size // self.patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        h, w = x.shape[-2:]

        x = x.flatten(2).transpose(1, 2)

        if x.shape[1] != self.position_embedding.shape[1]:
            position = self.position_embedding.transpose(1, 2)
            size = int(self.position_embedding.shape[1] ** 0.5)
            position = position.reshape(1, -1, size, size)
            position = F.interpolate(position, size=(h, w), mode="bilinear", align_corners=False)
            position = position.flatten(2).transpose(1, 2)
        else:
            position = self.position_embedding

        x = x + position
        x = self.dropout(x)

        return x, (h, w)


class TransformerBottleneck(nn.Module):
    """
    Transformer encoder operating on CNN bottleneck features.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        feature_size: int = 16,
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=1,
            feature_size=feature_size,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.reconstruct = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, spatial_size = self.patch_embed(x)
        tokens = self.transformer(tokens)

        h, w = spatial_size
        features = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], h, w)
        features = self.reconstruct(features)

        if features.shape[-2:] != x.shape[-2:]:
            features = F.interpolate(features, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return features + x


class TransUNet(nn.Module):
    """
    TransUNet-style segmentation model.

    Input:
        Tensor [B, C, H, W]

    Output:
        Logit mask [B, output_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 32,
        embed_dim: int = 256,
        transformer_layers: int = 12,
        attention_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        input_size: int = 256,
    ) -> None:
        super().__init__()

        self.input_size = int(input_size)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = ConvBlock(in_channels, c1, dropout=0.0)
        self.enc2 = DownBlock(c1, c2, dropout=0.0)
        self.enc3 = DownBlock(c2, c3, dropout=0.0)
        self.enc4 = DownBlock(c3, c4, dropout=dropout)
        self.bottleneck = DownBlock(c4, c5, dropout=dropout)

        feature_size = max(1, self.input_size // 16)

        self.transformer = TransformerBottleneck(
            in_channels=c5,
            embed_dim=embed_dim,
            num_layers=transformer_layers,
            num_heads=attention_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            feature_size=feature_size,
        )

        self.dec4 = UpBlock(c5, c4, c4, dropout=dropout)
        self.dec3 = UpBlock(c4, c3, c3, dropout=dropout)
        self.dec2 = UpBlock(c3, c2, c2, dropout=0.0)
        self.dec1 = UpBlock(c2, c1, c1, dropout=0.0)

        self.out_conv = nn.Conv2d(c1, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        x5 = self.bottleneck(x4)
        x5 = self.transformer(x5)

        d4 = self.dec4(x5, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        logits = self.out_conv(d1)

        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)

        return logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Return binary segmentation mask from model logits.
        """
        self.eval()
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()


class DiceBCELoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice loss for segmentation.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = torch.sum(probs * targets, dim=1)
        denominator = torch.sum(probs, dim=1) + torch.sum(targets, dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def build_transunet(config: Dict) -> TransUNet:
    """
    Build TransUNet model from config.yaml dictionary.
    """
    data_cfg = config.get("data", {})
    model_cfg = config.get("transunet", {})

    in_channels = int(data_cfg.get("in_channels", 1))
    output_channels = int(model_cfg.get("output_channels", 1))

    model = TransUNet(
        in_channels=in_channels,
        output_channels=output_channels,
        base_channels=int(model_cfg.get("base_channels", 32)),
        embed_dim=int(model_cfg.get("embed_dim", 256)),
        transformer_layers=int(model_cfg.get("transformer_layers", 12)),
        attention_heads=int(model_cfg.get("attention_heads", 8)),
        mlp_dim=int(model_cfg.get("mlp_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        input_size=int(model_cfg.get("input_size", data_cfg.get("image_size", 256))),
    )

    return model


def build_segmentation_loss(config: Dict) -> DiceBCELoss:
    """
    Build segmentation loss from config.yaml dictionary.
    """
    loss_cfg = config.get("loss", {}).get("segmentation", {})

    return DiceBCELoss(
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
    )


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable model parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


if __name__ == "__main__":
    dummy_config = {
        "data": {
            "in_channels": 1,
            "image_size": 256,
        },
        "transunet": {
            "output_channels": 1,
            "base_channels": 32,
            "embed_dim": 256,
            "transformer_layers": 2,
            "attention_heads": 8,
            "mlp_dim": 512,
            "dropout": 0.1,
            "input_size": 256,
        },
    }

    model = build_transunet(dummy_config)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Trainable parameters:", count_parameters(model))
