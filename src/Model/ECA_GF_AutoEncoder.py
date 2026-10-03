from itertools import chain
import torch
from torch import nn

from .AutoEncoder import Encoder, Decoder
from .ECA_Net import ECA_Net
from .Gated_Fusion import GatedFusionUnit


class LatentNoiseInjection(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            return x + self.sigma * torch.randn_like(x)
        return x


class ECA_GF_Autoencoder(nn.Module):
    def __init__(self, input_dim=40, output_dim=40,
                 hidden_neus=27, latent_dim=7,
                 eca_k_size=3, noise_sigma=0.1,
                 shrink_lambda=None):
        super(ECA_GF_Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_neus, latent_dim)
        self.eca = ECA_Net(channels=latent_dim, k_size=eca_k_size)
        self.gfu = GatedFusionUnit(feature_dim=latent_dim)
        self.decoder = Decoder(latent_dim, hidden_neus, output_dim)
        self.noise = LatentNoiseInjection(sigma=noise_sigma)
        self.shrink_lambda = shrink_lambda
        self.last_gate_weight = None

    def paramaeters(self):
        return chain(
            self.encoder.parameters(),
            self.eca.parameters(),
            self.gfu.parameters(),
            self.decoder.parameters(),
        )

    def recon_loss(self, input, output, latent=None):
        loss = nn.MSELoss(reduction="mean")(input, output)
        if self.shrink_lambda is not None and latent is not None:
            loss = loss + self.shrink_lambda * torch.mean(torch.norm(latent, dim=1))
        return loss

    def forward(self, input):
        if input.dim() != 2:
            raise ValueError(
                "ECA_GF_Autoencoder expects tabular input with shape (B, F), "
                f"but got {tuple(input.shape)}."
        )

        latent = self.encoder(input)
        latent_eca = self.eca(latent)
        fused_latent, gate_weight = self.gfu(latent, latent_eca)
        output = self.decoder(self.noise(fused_latent))
        loss = self.recon_loss(input, output, fused_latent)

        self.last_gate_weight = gate_weight.detach()
        return fused_latent, output, loss

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()


class ECA_AE_Autoencoder(nn.Module):
    def __init__(self, input_dim=40, output_dim=40,
                 hidden_neus=27, latent_dim=7,
                 eca_k_size=3):
        super(ECA_AE_Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_neus, latent_dim)
        self.eca = ECA_Net(channels=latent_dim, k_size=eca_k_size)
        self.decoder = Decoder(latent_dim, hidden_neus, output_dim)

    def paramaeters(self):
        return chain(
            self.encoder.parameters(),
            self.eca.parameters(),
            self.decoder.parameters(),
        )

    def recon_loss(self, input, output):
        return nn.MSELoss(reduction="mean")(input, output)

    def forward(self, input):
        latent = self.encoder(input)
        latent_eca = self.eca(latent)
        output = self.decoder(latent_eca)
        loss = self.recon_loss(input, output)
        return latent_eca, output, loss

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()


class GF_AE_Autoencoder(nn.Module):
    def __init__(self, input_dim=40, output_dim=40,
                 hidden_neus=27, latent_dim=7):
        super(GF_AE_Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_neus, latent_dim)
        self.latent_projection = nn.Linear(latent_dim, latent_dim, bias=True)
        self.gfu = GatedFusionUnit(feature_dim=latent_dim)
        self.decoder = Decoder(latent_dim, hidden_neus, output_dim)
        self.last_gate_weight = None

    def paramaeters(self):
        return chain(
            self.encoder.parameters(),
            self.latent_projection.parameters(),
            self.gfu.parameters(),
            self.decoder.parameters(),
        )

    def recon_loss(self, input, output):
        return nn.MSELoss(reduction="mean")(input, output)

    def forward(self, input):
        latent = self.encoder(input)
        latent_projected = self.latent_projection(latent)
        fused_latent, gate_weight = self.gfu(latent, latent_projected)
        output = self.decoder(fused_latent)
        loss = self.recon_loss(input, output)
        self.last_gate_weight = gate_weight.detach()
        return fused_latent, output, loss

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()
