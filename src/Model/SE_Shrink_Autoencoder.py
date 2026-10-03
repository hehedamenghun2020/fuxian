"""
Shrink Autoencoder with an input SE feature recalibration block.
"""

from itertools import chain
import numpy as np
import torch
from torch import nn


class SE_Block(nn.Module):
    def __init__(self, feature_dim, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.ReLU(),
            nn.Linear(feature_dim // reduction, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class Encoder(nn.Module):
    def __init__(self, input_dim=40, hidden_neus=27, latent_dim=7):
        super(Encoder, self).__init__()
        encoder_network = []
        encoder_network.append(nn.Linear(input_dim, hidden_neus, bias=True))
        encoder_network.append(nn.ReLU())
        encoder_network.append(nn.Linear(hidden_neus, latent_dim, bias=True))
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
    def __init__(self, latent_dim=7, hidden_neus=27, output_dim=40):
        super(Decoder, self).__init__()
        decoder_network = []
        decoder_network.append(nn.Linear(latent_dim, hidden_neus, bias=True))
        decoder_network.append(nn.ReLU())
        decoder_network.append(nn.Linear(hidden_neus, output_dim, bias=True))
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


class SE_Shrink_Autoencoder(nn.Module):
    def __init__(self, input_dim=40, output_dim=40,
                 hidden_neus=27, latent_dim=7,
                 shrink_lambda=10, reduction=16):
        super(SE_Shrink_Autoencoder, self).__init__()
        self.se = SE_Block(input_dim, reduction=reduction)
        self.encoder = Encoder(input_dim, hidden_neus, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_neus, output_dim)
        self.shrink_lambda = shrink_lambda

    def paramaeters(self):
        return chain(self.se.parameters(), self.encoder.parameters(), self.decoder.parameters())

    def shrink_loss(self, input, output, latent):
        batch_loss = nn.MSELoss(reduction='mean')(input, output) + \
            self.shrink_lambda * (torch.sum(torch.linalg.vector_norm(latent, dim=1)) / latent.shape[0])
        return batch_loss

    def forward(self, input):
        refined = self.se(input)
        latent = self.encoder(refined)
        output = self.decoder(latent)
        loss = self.shrink_loss(input, output, latent)
        return latent, output, loss

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()
