import torch.nn as nn
from torchvision.models import mobilenet_v2


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MobileNet_v2_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._mobilenetv2 = mobilenet_v2()
        self._mobilenetv2.classifier = Identity()

    def forward(self, input_img):
        return self._mobilenetv2(input_img)


class MobileNet_v2(nn.Module):
    def __init__(self, num_class) -> None:
        super().__init__()

        self._encoder = MobileNet_v2_Encoder()
        self._classifier = nn.Linear(1280, num_class)
        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self._encoder(x)
        out = self._classifier(emb)
        out = self._sigmoid(out)
        return out, emb
