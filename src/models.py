"""Column convention: W1 (H,D), W2 (O,H). Batches use x @ W.T."""

import torch
from torch import nn


class OriginalMLP(nn.Module):
    def __init__(self, d_in: int = 16, d_hidden: int = 48, d_out: int = 12):
        super().__init__()
        if min(d_in, d_hidden, d_out) <= 0:
            raise ValueError("All model dimensions must be positive")
        self.W1 = nn.Linear(d_in, d_hidden)
        self.W2 = nn.Linear(d_hidden, d_out)

    def hidden(self, x):
        if x.ndim != 2 or x.shape[1] != self.W1.in_features:
            raise ValueError("Expected x of shape (batch, d_in)")
        preactivation = self.W1(x)
        return preactivation, torch.relu(preactivation)

    def forward(self, x):
        return self.W2(self.hidden(x)[1])


class SparseTranscoder(nn.Module):
    """E (K,D), e (K), B (O,K), c (O). Input is x, never a or g."""

    def __init__(self, d_in, width, d_out):
        super().__init__()
        self.encoder = nn.Linear(d_in, width)
        self.decoder = nn.Linear(width, d_out)
        nn.init.constant_(self.encoder.bias, -0.1)
        self.normalize_decoder()

    def latents(self, x):
        return torch.relu(self.encoder(x))

    def forward(self, x):
        return self.decoder(self.latents(x))

    @torch.no_grad()
    def normalize_decoder(self):
        """Fix the positive-rescaling degeneracy of ReLU activation L1."""
        norms = self.decoder.weight.norm(dim=0, keepdim=True)
        if torch.any(norms == 0):
            raise RuntimeError("Zero decoder column cannot be normalized")
        self.decoder.weight.div_(norms)
