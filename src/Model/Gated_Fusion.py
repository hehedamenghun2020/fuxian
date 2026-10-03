import torch
import torch.nn as nn


class GatedFusionUnit(nn.Module):
    """
    Fuse original and attention-enhanced latent features with a learnable gate.
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, original_feature, enhanced_feature):
        if original_feature.shape != enhanced_feature.shape:
            raise ValueError(
                "GatedFusionUnit requires original_feature and enhanced_feature "
                f"to have the same shape, got {tuple(original_feature.shape)} vs "
                f"{tuple(enhanced_feature.shape)}."
            )

        gate_weight = self.gate(torch.cat([original_feature, enhanced_feature], dim=-1))
        fused_feature = gate_weight * enhanced_feature + (1.0 - gate_weight) * original_feature
        return fused_feature, gate_weight
