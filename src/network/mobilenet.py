import torch.nn as nn
from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MobileNet(nn.Module):
    def __init__(self, num_class: int = 1, type: str = "mobilenet_v2") -> None:
        super().__init__()
        if type == "mobilenet_v2":
            self._encoder = mobilenet_v2()
            self._encoder.classifier = Identity()
            self._classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=1280, out_features=10, bias=True),
            )
        elif type == "mobilenet_v3_small":
            self._encoder = mobilenet_v3_small()
            self._encoder.classifier = Identity()
            self._classifier = nn.Sequential(
                nn.Linear(in_features=576, out_features=1024, bias=True),
                nn.Hardswish(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1024, out_features=num_class, bias=True),
            )
        elif type == "mobilenet_v3_large":
            self._encoder = mobilenet_v3_large()
            self._encoder.classifier = Identity()
            self._classifier = nn.Sequential(
                nn.Linear(in_features=960, out_features=1280, bias=True),
                nn.Hardswish(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1280, out_features=num_class, bias=True),
            )

        self._sigmoid = nn.Sigmoid()

    def _get_feature(self, x):
        emb = self._encoder(x)
        return emb

    def _get_prediction(self, emb):
        out = self._classifier(emb)
        out = self._sigmoid(out)
        return out

    def forward(self, x):
        emb = self._encoder(x)
        out = self._classifier(emb)
        out = self._sigmoid(out)
        return out, emb
