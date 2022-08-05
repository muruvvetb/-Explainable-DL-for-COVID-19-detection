import torch as torch
import torchvision.models as models


class DeiTBase(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=self.pretrained)
        # 86m params
        self.out_size = self.model.head.out_features

    def forward(self, x):
        return self.model(x)


class DeiTTiny(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=self.pretrained)
        # 5.6m params
        self.out_size = self.model.head.out_features

    def forward(self, x):
        return self.model(x)


class DeiTSmall(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224',
                                    pretrained=self.pretrained)
        # 22m params
        self.out_size = self.model.head.out_features

    def forward(self, x):
        return self.model(x)
