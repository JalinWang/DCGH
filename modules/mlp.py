import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    # nn.Linear(input_dim, 512),
    # nn.ReLU(inplace=True),
    # nn.Linear(512, 1024),
    # nn.ReLU(inplace=True),
    # nn.Linear(1024, 128),
    # nn.ReLU(inplace=True),
    # nn.Linear(128, output_dim),
    # nn.Tanh()
    def forward(self, x):
        x = self.net(x)
        return x


class MLP_mod(nn.Module):
    def __init__(self, code_length):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(10, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.hash = nn.Sequential(
            nn.Linear(512, code_length),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.feature(x)
        hash = self.hash(y)
        return hash


def mlp(*args):
    model = MLP(*args)

    return model
