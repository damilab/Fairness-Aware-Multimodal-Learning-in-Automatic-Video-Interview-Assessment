import torch
import torch.nn as nn
from numpy.random import beta


# for single sensitive attribute
class MixUp_ERM(nn.Module):
    def __init__(self):
        super().__init__()
        self._base_loss = torch.nn.BCELoss()

    def _cal_mixup_erm(
        self,
        model: nn.Module,
        batch_image: torch.Tensor,
        batch_target: torch.Tensor,
    ):
        alpha = 1
        gamma = beta(alpha, alpha)

        batch_size = batch_image.size()[0]
        index = torch.randperm(batch_size)
        mixed_batch_image = gamma * batch_image + (1 - gamma) * batch_image[index, :]
        y_a, y_b = batch_target, batch_target[index]

        pred, _ = model(mixed_batch_image)

        loss = gamma * self._base_loss(pred, y_a) + (1 - gamma) * self._base_loss(
            pred, y_b
        )
        return loss

    def forward(self, model, batch_image, batch_target):
        return self._cal_mixup_erm(model, batch_image, batch_target)


# for single sensitive attribute
class MixUp_ERM_Group(nn.Module):
    def __init__(self):
        super().__init__()
        self._base_loss = torch.nn.BCELoss()

    def _cal_mixup_erm_group(
        self,
        model: nn.Module,
        batch_image: torch.Tensor,
        batch_group: torch.Tensor,
        batch_target: torch.Tensor,
    ):
        alpha = 1
        gamma = beta(alpha, alpha)

        inputs_0 = batch_image[torch.where(batch_group == 0)[0]]
        inputs_1 = batch_image[torch.where(batch_group == 1)[0]]
        min_len = min(inputs_0.shape[0], inputs_1.shape[0])
        inputs_0 = inputs_0[:min_len]
        inputs_1 = inputs_1[:min_len]

        targets_0 = batch_target[torch.where(batch_group == 0)[0]][:min_len]
        targets_1 = batch_target[torch.where(batch_group == 1)[0]][:min_len]

        mixed_batch_image = gamma * inputs_0 + (1 - gamma) * inputs_1
        y_a, y_b = targets_0, targets_1

        pred, _ = model(mixed_batch_image)

        loss = gamma * self._base_loss(pred, y_a) + (1 - gamma) * self._base_loss(
            pred, y_b
        )
        return loss

    def forward(self, model, batch_image, batch_group, batch_target):
        return self._cal_mixup_erm_group(model, batch_image, batch_group, batch_target)
