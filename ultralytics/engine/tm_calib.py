"""Teacher-Mask Calibration Distillation for ProtoHGF-Net.

The losses in this file implement Equations (9)-(14) of the ProtoHGF-Net
paper.  They are detector-agnostic: callers only need to provide matching
teacher and student feature maps.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


__all__ = ("TMCalibLoss", "MultiScaleTMCalibLoss")


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention used in Equation (9)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class SpatialGate(nn.Module):
    """Spatial attention from channel-average and channel-maximum maps."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        average = feature.mean(dim=1, keepdim=True)
        maximum = feature.amax(dim=1, keepdim=True)
        return self.layers(torch.cat((average, maximum), dim=1))


class TeacherMaskGuidedDecomposition(nn.Module):
    """Generate a joint teacher mask and foreground/background features."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialGate()
        self.refinement = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self, teacher_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        channel_mask = self.channel_attention(teacher_feature)
        spatial_mask = self.spatial_attention(teacher_feature)
        joint_mask = (channel_mask * spatial_mask * self.refinement(teacher_feature)).clamp(0.0, 1.0)
        foreground = joint_mask * teacher_feature
        background = teacher_feature - foreground
        return foreground, background, joint_mask, spatial_mask


class TMCalibLoss(nn.Module):
    """TM-Calib loss for one modality at one feature-pyramid level.

    The modality teacher itself is frozen by the training pipeline.  Teacher
    features are detached here as a second safeguard, while the lightweight
    mask generator remains learnable.
    """

    def __init__(
        self,
        channels: int,
        lambda_foreground: float = 1.0,
        lambda_background: float = 2.0,
        lambda_orthogonality: float = 0.07,
        attention_reduction: int = 16,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.decomposition = TeacherMaskGuidedDecomposition(channels, attention_reduction)
        self.lambda_foreground = lambda_foreground
        self.lambda_background = lambda_background
        self.lambda_orthogonality = lambda_orthogonality
        self.eps = eps

    def _foreground_weights(
        self,
        teacher_foreground: torch.Tensor,
        joint_mask: torch.Tensor,
        spatial_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reliability map W used by foreground alignment (Equation 11)."""
        batch = teacher_foreground.shape[0]
        response = teacher_foreground.detach().abs().mean(dim=1, keepdim=True)
        maximum = response.flatten(1).amax(dim=1).reshape(batch, 1, 1, 1).clamp_min(self.eps)
        response = response / maximum
        channel_support = joint_mask.detach().mean(dim=1, keepdim=True)
        return (response * channel_support * spatial_mask.detach()).clamp(0.0, 1.0)

    def forward(
        self,
        teacher_feature: torch.Tensor,
        student_feature: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if teacher_feature.shape != student_feature.shape:
            raise ValueError(
                "Teacher and student features must have identical shapes, got "
                f"{tuple(teacher_feature.shape)} and {tuple(student_feature.shape)}"
            )

        teacher_feature = teacher_feature.detach()
        teacher_fg, _, joint_mask, spatial_mask = self.decomposition(teacher_feature)
        student_fg = joint_mask * student_feature
        student_bg = student_feature - student_fg

        weights = self._foreground_weights(teacher_fg, joint_mask, spatial_mask)
        squared_error = (student_fg - teacher_fg).square().sum(dim=1, keepdim=True)
        foreground_loss = (weights * squared_error).sum() / (weights.sum() + self.eps)

        background_weights = (1.0 - spatial_mask).clamp(0.0, 1.0)
        background_energy = student_bg.square().mean(dim=1, keepdim=True)
        background_loss = (background_weights * background_energy).sum() / (
            background_weights.sum() + self.eps
        )

        batch = student_feature.shape[0]
        foreground_vector = student_fg.reshape(batch, -1)
        background_vector = student_bg.reshape(batch, -1)
        orthogonality_loss = F.cosine_similarity(
            foreground_vector, background_vector, dim=1, eps=self.eps
        ).abs().mean()

        total = (
            self.lambda_foreground * foreground_loss
            + self.lambda_background * background_loss
            + self.lambda_orthogonality * orthogonality_loss
        )
        if not return_components:
            return total
        return total, {
            "foreground": foreground_loss.detach(),
            "background": background_loss.detach(),
            "orthogonality": orthogonality_loss.detach(),
        }


class MultiScaleTMCalibLoss(nn.Module):
    """Sum TM-Calib over RGB/Thermal modalities and feature-pyramid levels."""

    def __init__(
        self,
        channels: Sequence[int] = (192, 384, 576),
        level_weights: Sequence[float] | None = None,
        lambda_foreground: float = 1.0,
        lambda_background: float = 2.0,
        lambda_orthogonality: float = 0.07,
        attention_reduction: int = 16,
    ):
        super().__init__()
        if level_weights is None:
            level_weights = (1.0,) * len(channels)
        if len(channels) != len(level_weights):
            raise ValueError("channels and level_weights must have the same length")
        self.level_weights = tuple(float(weight) for weight in level_weights)
        self.level_losses = nn.ModuleList(
            TMCalibLoss(
                channel,
                lambda_foreground=lambda_foreground,
                lambda_background=lambda_background,
                lambda_orthogonality=lambda_orthogonality,
                attention_reduction=attention_reduction,
            )
            for channel in channels
        )

    def forward(
        self,
        teacher_rgb: Sequence[torch.Tensor],
        student_rgb: Sequence[torch.Tensor],
        teacher_thermal: Sequence[torch.Tensor],
        student_thermal: Sequence[torch.Tensor],
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected = len(self.level_losses)
        feature_groups = (teacher_rgb, student_rgb, teacher_thermal, student_thermal)
        if any(len(features) != expected for features in feature_groups):
            lengths = tuple(len(features) for features in feature_groups)
            raise ValueError(f"Expected {expected} feature levels, received lengths {lengths}")

        total = student_rgb[0].new_zeros(())
        component_totals = {
            "foreground": student_rgb[0].new_zeros(()),
            "background": student_rgb[0].new_zeros(()),
            "orthogonality": student_rgb[0].new_zeros(()),
        }
        for index, (level_loss, weight) in enumerate(zip(self.level_losses, self.level_weights)):
            rgb_loss, rgb_parts = level_loss(
                teacher_rgb[index], student_rgb[index], return_components=True
            )
            thermal_loss, thermal_parts = level_loss(
                teacher_thermal[index], student_thermal[index], return_components=True
            )
            total = total + weight * (rgb_loss + thermal_loss)
            for name in component_totals:
                component_totals[name] = component_totals[name] + weight * (
                    rgb_parts[name] + thermal_parts[name]
                )

        if return_components:
            return total, component_totals
        return total
