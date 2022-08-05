import torch as torch
from collections import OrderedDict


class UnFlatten(torch.nn.Module):
    def forward(self, x, size=1000):
        return x.view(x.size(0), size, 1, 1)


class Decoder(torch.nn.Module):
    def __init__(self, mode=None):
        super().__init__()

        if mode == "up":
            mode = "bilinear"
            self.decoder = torch.nn.Sequential(
                UnFlatten(),
                torch.nn.Upsample(scale_factor=7, mode=mode),
                torch.nn.Conv2d(1000, 512, kernel_size=(3,3), stride=(1, 1), padding=(1, 1)),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode=mode),
                torch.nn.Conv2d(512, 256, kernel_size=(3,3), stride=(1, 1), padding=(1, 1)),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode=mode),
                torch.nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1, 1), padding=(1, 1)),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode=mode),
                torch.nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1, 1), padding=(1, 1)),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode=mode),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1, 1), padding=(1, 1)),
                torch.nn.Upsample(scale_factor=2, mode=mode),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 1, kernel_size=(3,3), stride=(1, 1), padding=(1, 1)),
                torch.nn.Sigmoid()
            )
        else:
            self.decoder = torch.nn.Sequential(
                UnFlatten(),
                torch.nn.ConvTranspose2d(1000, 512, kernel_size=(5, 5), stride=(2, 2)),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2)),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2)),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2)),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2)),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(32, 1, kernel_size=(6, 6), stride=(2, 2)),
                torch.nn.Sigmoid()
            )

    def forward(self, x):
        return self.decoder(x)


