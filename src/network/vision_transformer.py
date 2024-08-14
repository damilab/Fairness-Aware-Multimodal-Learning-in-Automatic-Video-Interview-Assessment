import torch
import torch.nn as nn
from torchvision.models import vit_b_16


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class VisionTransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vit_encoder = vit_b_16()
        vit_encoder.heads = Identity()
        self.model = vit_encoder

    def forward(self, input_img):
        return self.model(input_img)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_class: int,
    ) -> None:
        super().__init__()

        self._encoder = VisionTransformerEncoder()
        self._classifier = nn.Linear(768, num_class)
        self._sigmoid = nn.Sigmoid()

    def _get_feature(self, x):
        emb = self._encoder(x)
        return emb

    def _get_prediction(self, emb):
        out = self._classifier(emb)
        out = self._sigmoid(out)
        return out

    def forward(self, x):
        emb = self._get_feature(x)
        out = self._get_prediction(emb)
        return out, emb
