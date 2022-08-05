import torch as torch
from architectures.encoder_bases.DeiT import DeiTTiny, DeiTSmall, DeiTBase
from architectures.encoder_bases.ResNet import ResNet18, ResNet50, ResNet101, ResNet152
from architectures.Classifier import Classifier
from architectures.Decoder import Decoder
import copy

encoders_dict = {
    "DeiTTiny": DeiTTiny,
    "DeiTSmall": DeiTSmall,
    "DeiTBase": DeiTBase,
    "ResNet18": ResNet18,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152
}

class DecoderEncoderDecoderOriginal(torch.nn.Module):
    def __init__(self, num_classes, encoder_type, hidden_size=64, dropout_rate=0.5, pretrained=True, decoder_mode="transpose"):
        super().__init__()

        assert encoder_type in ["DeiTTiny", "DeiTSmall", "DeiTBase", "ResNet18", "ResNet50", "ResNet101",
                                "ResNet152"], f"NotImplemented Base {encoder_type}"

        self.encoder_type = encoder_type
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.decoder_mode = decoder_mode

        self.encoder = encoders_dict[self.encoder_type](self.pretrained)
        self.classifier = Classifier(in_size=self.encoder.out_size, out_size=self.num_classes,
                                     hidden_size=self.hidden_size, dropout_rate=self.dropout_rate)
        self.decoder = Decoder(mode=self.decoder_mode)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)

        explanations = self.decoder(features)
        repeated_explanations = explanations.repeat(1, 3, 1, 1)
        # x_masked = x.detach().clone()
        # x_masked[(explanations < 1).repeat(1, 3, 1, 1)] *= 0.75
        #  image_and_explanations =  x_masked# 0.25*x + 0.75*repeated_explanations
        image_and_explanations = repeated_explanations
        features_explanations = self.encoder(image_and_explanations)
        logits_explanations = self.classifier(features_explanations)

        return logits, logits_explanations, explanations, image_and_explanations

    def predict(self, x):
        logits, logits_explanations, explanations, image_and_explanations = self.forward(x)
        classes = torch.nn.Softmax(dim=1)(logits).argmax(dim=1)
        classes_explanations = torch.nn.Softmax(dim=1)(logits_explanations).argmax(dim=1)
        return classes, classes_explanations, explanations

    def get_probabilities(self, x):
        logits, _, _, _ = self.forward(x)
        return torch.nn.Softmax(dim=1)(logits)

    def loss(self, x, y, epoch=None):
        logits, logits_explanations, explanations, image_and_explanations = self.forward(x)

        loss1 = torch.nn.CrossEntropyLoss()(logits, y)
        loss2 = torch.nn.CrossEntropyLoss()(logits_explanations, y)

        return {"loss": loss1 + loss2, "logits": logits, "logits_explanations": logits_explanations,
                "explanations": explanations}

class DecoderEncoderDecoder(torch.nn.Module):
    def __init__(self, num_classes, encoder_type, hidden_size=64, dropout_rate=0.5, pretrained=True, decoder_mode="transpose"):
        super().__init__()

        assert encoder_type in ["DeiTTiny", "DeiTSmall", "DeiTBase", "ResNet18", "ResNet50", "ResNet101",
                                "ResNet152"], f"NotImplemented Base {encoder_type}"

        self.encoder_type = encoder_type
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.decoder_mode = decoder_mode

        self.encoder = encoders_dict[self.encoder_type](self.pretrained)
        self.classifier = Classifier(in_size=self.encoder.out_size, out_size=self.num_classes,
                                     hidden_size=self.hidden_size, dropout_rate=self.dropout_rate)
        self.decoder = Decoder(mode=self.decoder_mode)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)

        explanations = self.decoder(features)
        repeated_explanations = explanations.repeat(1, 3, 1, 1)
        # x_masked = x.detach().clone()
        # x_masked[(explanations < 1).repeat(1, 3, 1, 1)] *= 0.75
        #  image_and_explanations =  x_masked# 0.25*x + 0.75*repeated_explanations
        image_and_explanations = x + repeated_explanations
        features_explanations = self.encoder(image_and_explanations)
        logits_explanations = self.classifier(features_explanations)

        return logits, logits_explanations, explanations, image_and_explanations

    def predict(self, x):
        logits, logits_explanations, explanations, image_and_explanations = self.forward(x)
        classes = torch.nn.Softmax(dim=1)(logits).argmax(dim=1)
        classes_explanations = torch.nn.Softmax(dim=1)(logits_explanations).argmax(dim=1)
        return classes, classes_explanations, explanations

    def get_probabilities(self, x):
        logits, _, _, _ = self.forward(x)
        return torch.nn.Softmax(dim=1)(logits)

    def loss(self, x, y, epoch=None):
        logits, logits_explanations, explanations, image_and_explanations = self.forward(x)

        loss1 = torch.nn.CrossEntropyLoss()(logits, y)
        loss2 = torch.nn.CrossEntropyLoss()(logits_explanations, y)

        return {"loss": loss1 + loss2, "logits": logits, "logits_explanations": logits_explanations,
                "explanations": explanations}


class DecoderEncoderDecoderv2(torch.nn.Module):
    def __init__(self, num_classes, encoder_type, hidden_size=64, dropout_rate=0.5, pretrained=True, decoder_mode="transpose"):
        super().__init__()

        assert encoder_type in ["DeiTTiny", "DeiTSmall", "DeiTBase", "ResNet18", "ResNet50", "ResNet101",
                                "ResNet152"], f"NotImplemented Base {encoder_type}"

        self.encoder_type = encoder_type
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.decoder_mode = decoder_mode

        self.encoder = encoders_dict[self.encoder_type](self.pretrained)
        self.classifier = Classifier(in_size=self.encoder.out_size, out_size=self.num_classes,
                                     hidden_size=self.hidden_size, dropout_rate=self.dropout_rate)

        self.encoder2 = encoders_dict[self.encoder_type](self.pretrained)
        self.classifier2 = Classifier(in_size=self.encoder.out_size, out_size=self.num_classes,
                                     hidden_size=self.hidden_size, dropout_rate=self.dropout_rate)

        self.decoder = Decoder(mode=self.decoder_mode)

    def forward(self, x, epoch):
        features = self.encoder(x)
        logits = self.classifier(features)
        if epoch < 25:
            return logits, torch.zeros_like(logits, device=logits.device), None, None

        explanations = self.decoder(features)
        repeated_explanations = explanations.repeat(1, 3, 1, 1)
        # x_masked = x.detach().clone()
        # x_masked[(explanations < 1).repeat(1, 3, 1, 1)] *= 0.75
        #  image_and_explanations =  x_masked# 0.25*x + 0.75*repeated_explanations
        image_and_explanations = x + repeated_explanations
        features_explanations = self.encoder2(image_and_explanations)
        logits_explanations = self.classifier2(features_explanations)

        return logits, logits_explanations, explanations, image_and_explanations

    def predict(self, x, epoch):
        logits, logits_explanations, explanations, image_and_explanations = self.forward(x, epoch)
        classes = torch.nn.Softmax(dim=1)(logits).argmax(dim=1)
        classes_explanations = torch.nn.Softmax(dim=1)(logits_explanations).argmax(dim=1)
        return classes, classes_explanations, explanations

    def get_probabilities(self, x):
        logits, _, _, _ = self.forward(x)
        return torch.nn.Softmax(dim=1)(logits)

    def loss(self, x, y, epoch=None):
        logits, logits_explanations, explanations, image_and_explanations = self.forward(x, epoch)

        loss1 = torch.nn.CrossEntropyLoss()(logits, y)
        
        if epoch == 25:
            self.encoder2 = copy.deepcopy(self.encoder)
            self.classifier2 = copy.deepcopy(self.classifier)

        if epoch < 25:
            loss2 = 0
        else:
            loss2 = torch.nn.CrossEntropyLoss()(logits_explanations, y)



        return {"loss": loss1 + loss2, "logits": logits, "logits_explanations": logits_explanations,
                "explanations": explanations}

