import torch.nn as nn


class ECA_Net(nn.Module):
    """
    Feature-wise ECA for tabular latent vectors.

    Supported input shapes:
    - (batch, features)
    - (batch, seq_len, features)
    """

    def __init__(self, channels=11, k_size=3):
        super().__init__()
        if k_size % 2 == 0:
            raise ValueError("k_size must be odd in ECA_Net.")
        self.channels = channels
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def _compute_weights(self, feature_tensor):
        attn = self.conv(feature_tensor.unsqueeze(1))
        attn = self.sigmoid(attn).squeeze(1)
        return attn

    def forward(self, x):
        if x.dim() == 2:
            weights = self._compute_weights(x)
            return x * weights

        if x.dim() == 3:
            pooled = x.mean(dim=1)
            weights = self._compute_weights(pooled)
            return x * weights.unsqueeze(1)

        raise ValueError(f"ECA_Net expects 2D or 3D input, but got shape: {tuple(x.shape)}")
