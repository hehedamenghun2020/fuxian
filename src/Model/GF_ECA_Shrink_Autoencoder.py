"""
GF-ECA-SAE model for tabular N-BaIoT features.
"""

import numpy as np
import torch
from torch import nn

from .ECA_Net import ECA_Net
from .Gated_Fusion import GatedFusionUnit


class Encoder(nn.Module):
    def __init__(self, input_dim=115, hidden_neus=50, latent_dim=11):
        super().__init__()
        encoder_network = [
            nn.Linear(input_dim, hidden_neus, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_neus, latent_dim, bias=True),
        ]
        self.encoder_network = nn.Sequential(*encoder_network)
        self.init_params()

    def init_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                bound = 1 / np.sqrt(layer.in_features)
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.zero_()

    def forward(self, inputs):
        return self.encoder_network(inputs)


class Decoder(nn.Module):
    def __init__(self, latent_dim=11, hidden_neus=50, output_dim=115):
        super().__init__()
        decoder_network = [
            nn.Linear(latent_dim, hidden_neus, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_neus, output_dim, bias=True),
        ]
        self.decoder_network = nn.Sequential(*decoder_network)
        self.init_params()

    def init_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                bound = 1 / np.sqrt(layer.in_features)
                layer.weight.data.uniform_(-bound, bound)
                layer.bias.data.zero_()

    def forward(self, latent):
        return self.decoder_network(latent)


class GF_ECA_Shrink_Autoencoder(nn.Module):
    """
    input -> encoder -> ECA latent recalibration -> gated fusion -> decoder
    """

    def __init__(self, input_dim=115, output_dim=115, hidden_neus=50, latent_dim=11, shrink_lambda=10):
        super().__init__()
        self.shrink_lambda = shrink_lambda

        self.encoder = Encoder(input_dim, hidden_neus, latent_dim)
        self.eca = ECA_Net(channels=latent_dim, k_size=3)
        self.gfu = GatedFusionUnit(feature_dim=latent_dim)
        self.decoder = Decoder(latent_dim, hidden_neus, output_dim)

    def _to_tabular_features(self, input_tensor):
        if input_tensor.dim() == 2:
            return input_tensor
        if input_tensor.dim() == 3:
            return input_tensor.mean(dim=1)
        raise ValueError(
            "GF_ECA_Shrink_Autoencoder expects 2D tabular input (B, F) or "
            f"legacy 3D input (B, L, F), but got {tuple(input_tensor.shape)}."
        )

    def shrink_loss(self, input_feature, output_feature, latent_feature):
        mse = nn.MSELoss(reduction="mean")(input_feature, output_feature)
        shrink = self.shrink_lambda * (
            torch.sum(torch.linalg.vector_norm(latent_feature, dim=1)) / latent_feature.shape[0]
        )
        return mse + shrink

    def forward(self, input_tensor):
        feature = self._to_tabular_features(input_tensor)
        latent = self.encoder(feature)
        latent_enhanced = self.eca(latent)
        fused_latent, _ = self.gfu(latent, latent_enhanced)
        output = self.decoder(fused_latent)
        loss = self.shrink_loss(feature, output, fused_latent)
        return fused_latent, output, loss
