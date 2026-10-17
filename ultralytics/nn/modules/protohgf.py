"""Prototype HyperGraph Fusion used by ProtoHGF-Net.

This file is the release-oriented implementation of Equations (1)-(8) in
"ProtoHGF-Net: Prototype HyperGraph Fusion with Intra-modal Calibration for
RGBT Object Detection".  The implementation deliberately keeps the graph
sparse and binary; the soft-relation ablation is not used here.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


__all__ = ("ProtoHGFusion",)


class MeanMessageAggregation(nn.Module):
    """Aggregate source nodes into target nodes with degree-normalized means."""

    def forward(self, source: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        """Apply ``relation[target, source]`` to a batch of source features."""
        degree = relation.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(relation, source) / degree


class HypergraphPropagation(nn.Module):
    """Two-step vertex -> hyperedge -> vertex propagation (Equation 5)."""

    def __init__(self, channels: int):
        super().__init__()
        self.projection = nn.Linear(channels, channels)
        self.aggregate = MeanMessageAggregation()

    def forward(self, vertices: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        vertices = self.projection(vertices)
        hyperedges = self.aggregate(vertices, incidence.transpose(1, 2).contiguous())
        return self.aggregate(hyperedges, incidence)


class PrototypeExtractor(nn.Module):
    """Compress a feature map into K spatially weighted semantic prototypes."""

    def __init__(self, channels: int, num_prototypes: int):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.attention = nn.Conv2d(channels, num_prototypes, kernel_size=1, bias=True)

    def forward(
        self, feature: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = feature.shape
        weights = self.attention(feature).reshape(batch, self.num_prototypes, -1)
        weights = weights.softmax(dim=-1)
        prototypes = torch.einsum("bkn,bcn->bkc", weights, feature.reshape(batch, channels, -1))
        if return_attention:
            return prototypes, weights.reshape(batch, self.num_prototypes, height, width)
        return prototypes


class GlobalModalityGate(nn.Module):
    """Predict one RGB/Thermal weight pair per sample (Equation 8)."""

    def __init__(self, channels: int):
        super().__init__()
        self.rgb_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.thermal_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        hidden = max(channels // 2, 16)
        self.predictor = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(
        self, rgb: torch.Tensor, thermal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_descriptor = F.adaptive_avg_pool2d(self.rgb_transform(rgb), 1).flatten(1)
        thermal_descriptor = F.adaptive_avg_pool2d(self.thermal_transform(thermal), 1).flatten(1)
        weights = self.predictor(torch.cat((rgb_descriptor, thermal_descriptor), dim=1)).softmax(dim=1)
        fused = weights[:, 0, None, None, None] * rgb + weights[:, 1, None, None, None] * thermal
        return fused, weights


class ProtoHGFusion(nn.Module):
    """Prototype-level sparse hypergraph fusion for paired RGB/Thermal features.

    Args:
        channels: Number of channels in each input feature map. It is injected by
            the Ultralytics model parser when this module is used in a YAML file.
        num_prototypes: Number K of semantic prototypes per modality.
        k_intra: Number of intra-modal nearest neighbours.
        k_cross: Number of cross-modal nearest neighbours.
        propagation_scale: Residual scale for propagated prototype features.
        detach_graph: Build hard relations from detached prototypes when True.
    """

    def __init__(
        self,
        channels: int,
        num_prototypes: int = 6,
        k_intra: int = 3,
        k_cross: int = 3,
        propagation_scale: float = 1.0,
        detach_graph: bool = False,
    ):
        super().__init__()
        if channels <= 0 or num_prototypes <= 0:
            raise ValueError("channels and num_prototypes must be positive")
        if k_intra <= 0 or k_cross <= 0:
            raise ValueError("k_intra and k_cross must be positive")

        self.channels = channels
        self.num_prototypes = num_prototypes
        self.k_intra = k_intra
        self.k_cross = k_cross
        self.propagation_scale = propagation_scale
        self.detach_graph = detach_graph

        self.rgb_prototypes = PrototypeExtractor(channels, num_prototypes)
        self.thermal_prototypes = PrototypeExtractor(channels, num_prototypes)
        self.propagation = HypergraphPropagation(channels)

        prototype_dim = num_prototypes * channels
        self.rgb_gamma = nn.Linear(prototype_dim, channels)
        self.rgb_beta = nn.Linear(prototype_dim, channels)
        self.thermal_gamma = nn.Linear(prototype_dim, channels)
        self.thermal_beta = nn.Linear(prototype_dim, channels)
        self.gate = GlobalModalityGate(channels)

    @staticmethod
    def _hard_topk(similarity: torch.Tensor, k: int) -> torch.Tensor:
        """Convert a similarity matrix into a row-wise binary top-k relation."""
        k = min(k, similarity.shape[-1])
        indices = similarity.topk(k=k, dim=-1).indices
        relation = torch.zeros_like(similarity)
        return relation.scatter_(dim=-1, index=indices, value=1.0)

    def build_relation_matrix(
        self, rgb_prototypes: torch.Tensor, thermal_prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Construct the block relation matrix H from Equations (2)-(4)."""
        rgb = F.normalize(rgb_prototypes, dim=-1)
        thermal = F.normalize(thermal_prototypes, dim=-1)
        batch, count, _ = rgb.shape

        rgb_intra = self._hard_topk(torch.matmul(rgb, rgb.transpose(1, 2)), self.k_intra)
        thermal_intra = self._hard_topk(
            torch.matmul(thermal, thermal.transpose(1, 2)), self.k_intra
        )
        identity = torch.eye(count, device=rgb.device, dtype=rgb.dtype).expand(batch, -1, -1)
        rgb_intra = torch.maximum(rgb_intra, identity)
        thermal_intra = torch.maximum(thermal_intra, identity)

        rgb_thermal = self._hard_topk(
            torch.matmul(rgb, thermal.transpose(1, 2)), self.k_cross
        )
        relation = rgb.new_zeros(batch, 2 * count, 2 * count)
        relation[:, :count, :count] = rgb_intra
        relation[:, count:, count:] = thermal_intra
        relation[:, :count, count:] = rgb_thermal
        relation[:, count:, :count] = rgb_thermal.transpose(1, 2)
        return relation

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor], return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(x) != 2:
            raise ValueError(f"ProtoHGFusion expects [RGB, Thermal], received {len(x)} inputs")
        rgb_feature, thermal_feature = x
        if rgb_feature.shape != thermal_feature.shape:
            raise ValueError(
                "RGB and Thermal features must have identical shapes, got "
                f"{tuple(rgb_feature.shape)} and {tuple(thermal_feature.shape)}"
            )
        if rgb_feature.ndim != 4 or rgb_feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected BCHW features with C={self.channels}, got {tuple(rgb_feature.shape)}"
            )

        rgb_prototypes, rgb_attention = self.rgb_prototypes(rgb_feature, return_attention=True)
        thermal_prototypes, thermal_attention = self.thermal_prototypes(
            thermal_feature, return_attention=True
        )
        graph_rgb = rgb_prototypes.detach() if self.detach_graph else rgb_prototypes
        graph_thermal = thermal_prototypes.detach() if self.detach_graph else thermal_prototypes
        relation = self.build_relation_matrix(graph_rgb, graph_thermal)

        prototypes = torch.cat((rgb_prototypes, thermal_prototypes), dim=1)
        propagated = prototypes + self.propagation_scale * self.propagation(prototypes, relation)
        rgb_updated, thermal_updated = propagated.split(self.num_prototypes, dim=1)

        batch = rgb_feature.shape[0]
        rgb_flat = rgb_updated.flatten(1)
        thermal_flat = thermal_updated.flatten(1)
        rgb_gamma = self.rgb_gamma(rgb_flat).reshape(batch, self.channels, 1, 1)
        rgb_beta = self.rgb_beta(rgb_flat).reshape(batch, self.channels, 1, 1)
        thermal_gamma = self.thermal_gamma(thermal_flat).reshape(batch, self.channels, 1, 1)
        thermal_beta = self.thermal_beta(thermal_flat).reshape(batch, self.channels, 1, 1)

        rgb_calibrated = rgb_feature * (1.0 + rgb_gamma) + rgb_beta
        thermal_calibrated = thermal_feature * (1.0 + thermal_gamma) + thermal_beta
        fused, modality_weights = self.gate(rgb_calibrated, thermal_calibrated)

        if not return_aux:
            return fused
        return fused, {
            "relation": relation,
            "rgb_attention": rgb_attention,
            "thermal_attention": thermal_attention,
            "modality_weights": modality_weights,
        }
