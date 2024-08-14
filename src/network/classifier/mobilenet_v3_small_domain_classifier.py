import torch
import torch.nn as nn
from network.layers import GradReverse


class MobileNetV3SmallDomainClassifier(nn.Module):
    def __init__(
        self,
        num_class: int = 1,
    ):
        super().__init__()
        self._classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=num_class, bias=True),
        )
        self._sigmoid = nn.Sigmoid()

    def update_lambd(self, lambd):
        self.lambd = lambd
        GradReverse.lambd = self.lambd

    def forward(self, feature):
        feature = GradReverse.apply(feature)
        out = self._classifier(feature)
        out = self._sigmoid(out)
        return out
