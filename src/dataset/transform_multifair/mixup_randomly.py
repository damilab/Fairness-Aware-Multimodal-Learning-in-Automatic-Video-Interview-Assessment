import torch
import torch.nn as nn
from numpy.random import beta


class MixupRandomly(nn.Module):
    def __init__(self):
        super().__init__()

    def _mixup_randomly(
        self,
        batch_image: torch.Tensor,
        batch_target: torch.Tensor,
        batch_group: torch.Tensor,
    ):
        batch_size = batch_group.shape[0]
        lam = beta(1.0, 1.0)
        index = torch.randperm(batch_size)
        inputs_mix = lam * batch_image + (1 - lam) * batch_image[index, :]
        targets_mix = lam * batch_target + (1 - lam) * batch_target[index, :]

        return inputs_mix, targets_mix

    def forward(self, batch_image, batch_target, batch_group):
        return self._mixup_randomly(batch_image, batch_target, batch_group)
