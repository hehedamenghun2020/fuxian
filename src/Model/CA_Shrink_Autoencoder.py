"""
Shrink Autoencoder with Center Alignment Loss.
"""

from itertools import chain
import numpy as np
import torch
from torch import nn


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


class CA_Shrink_Autoencoder(nn.Module):
    def __init__(self, input_dim=40, output_dim=40,
                 hidden_neus=27, latent_dim=7,
                 shrink_lambda=10, beta=1.0):
        super(CA_Shrink_Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_neus, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_neus, output_dim)
        self.shrink_lambda = shrink_lambda
        self.beta = beta
        self.global_center = None

    def paramaeters(self):
        return chain(self.encoder.parameters(), self.decoder.parameters())

    def set_global_center(self, center):
        if center is not None:
            self.global_center = center.detach().clone()

    def shrink_loss(self, input, output, latent):
        mse = nn.MSELoss(reduction='mean')(input, output)
        shrink = self.shrink_lambda * (
            torch.sum(torch.linalg.vector_norm(latent, dim=1)) / latent.shape[0]
        )
        batch_center = latent.mean(dim=0)
        if self.global_center is not None:
            center_loss = torch.sum((batch_center - self.global_center) ** 2)
        else:
            center_loss = torch.sum(batch_center ** 2)
        return mse + shrink + self.beta * center_loss

    def forward(self, input):
        latent = self.encoder(input)
        output = self.decoder(latent)
        loss = self.shrink_loss(input, output, latent)
        return latent, output, loss

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()
