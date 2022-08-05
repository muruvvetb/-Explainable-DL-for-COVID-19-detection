import torch as torch
import torchvision.models as models


class ResNet18(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.resnet18(pretrained=self.pretrained)
        # 11.689512m parameters
        self.out_size = self.model.fc.out_features

    def forward(self, x):
        return self.model(x)


class ResNet50(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.resnet50(pretrained=self.pretrained)
        # 25.557032m parameters
        self.out_size = self.model.fc.out_features

    def forward(self, x):
        return self.model(x)


class ResNet101(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.resnet101(pretrained=self.pretrained)
        # 44.54916m parameters
        self.out_size = self.model.fc.out_features

    def forward(self, x):
        return self.model(x)


class ResNet152(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.resnet152(pretrained=self.pretrained)
        # 60.192m parameters
        self.out_size = self.model.fc.out_features

    def forward(self, x):
        return self.model(x)
