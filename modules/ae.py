from torch import nn
import torch
from torch.nn import functional as F


class AE(nn.Module):
    def __init__(self, code_length, hidden_size, embedding_size):
        super().__init__()

        self.code_length = code_length
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        # self.encoder = nn.Linear(code_length, embedding_size)
        # self.decoder = nn.Linear(embedding_size, code_length)
        self.encoder = nn.Sequential(
            nn.Linear(code_length, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, embedding_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, code_length)
        )

    def forward(self, x):
        latent = self.encoder(x)
        # rec_x = torch.tanh(self.decoder(latent))
        rec_x = (self.decoder(latent))
        return rec_x, latent


def ae(*args):
    model = AE(*args)

    return model
