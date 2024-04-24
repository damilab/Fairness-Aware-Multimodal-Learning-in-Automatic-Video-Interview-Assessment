import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ResNetEncoder(nn.Module):
    def __init__(
        self, pretrained_path: str = "None", freeze_encoder: bool = False, type="res18"
    ):
        super().__init__()

        if type == "res18":
            resnet = resnet18()

        elif type == "res34":
            resnet = resnet34()

        if pretrained_path != "None":
            weight_dict = self._load_pretrained(pretrained_path)
            resnet.load_state_dict(weight_dict, strict=False)

        if freeze_encoder:
            for param in resnet.parameters():
                param.requires_grad = False

        resnet.fc = Identity()

        self.model = resnet

    def forward(self, input_img):
        return self.model(input_img)

    def _load_pretrained(self, pretrained_path):
        weight_dict = {}
        weights = torch.load(pretrained_path)["state_dict"]
        for key, value in weights.items():
            if key.split(".")[1] == "fc" or key.split(".")[1] == "feature":
                continue
            weight_dict[key.replace("module.", "")] = value
        return weight_dict


class ResNet(nn.Module):
    def __init__(
        self, num_class, pretrained_path: str = "None", type="resnet18"
    ) -> None:
        super().__init__()

        self._encoder = ResNetEncoder(type=type, pretrained_path=pretrained_path)
        self._classifier = nn.Linear(512, num_class)
        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self._encoder(x)
        out = self._classifier(emb)
        out = self._sigmoid(out)
        return out, emb


class Classifier(nn.Module):
    def __init__(self, num_class = 1) -> None:
        super().__init__()

        self._classifier = nn.Linear(512, num_class)
        self._sigmoid = nn.Sigmoid()

    def forward(self, emb):
        out = self._classifier(emb)
        out = self._sigmoid(out)
        return out