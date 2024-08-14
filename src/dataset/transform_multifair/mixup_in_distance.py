import torch
import torch.nn as nn
from .mixup import Mixup


class MixupInDistance(nn.Module):
    def __init__(self, num_group: int):
        super().__init__()
        self._num_group = num_group
        self._mixup = Mixup()

    def _mixup_in_distance(
        self,
        batch_image: torch.Tensor,
        batch_target: torch.Tensor,
        batch_group: torch.Tensor,
    ):
        inputs_mix_list = []
        targets_mix_list = []
        for i in range(self._num_group // 2):
            j = self._num_group - 1 - i
            group_image_0 = batch_image[torch.where(batch_group == i)[0]]
            group_target_0 = batch_target[torch.where(batch_group == i)[0]]
            group_image_1 = batch_image[torch.where(batch_group == j)[0]]
            group_target_1 = batch_target[torch.where(batch_group == j)[0]]

            mixup_results = self._mixup._mixup_once(
                image_list=[group_image_0, group_image_1],
                target_list=[group_target_0, group_target_1],
            )
            inputs_mix_list.append(mixup_results["inputs_mix"])
            targets_mix_list.append(mixup_results["targets_mix"])

        inputs_mix = torch.cat(inputs_mix_list, dim=0)
        targets_mix = torch.cat(targets_mix_list, dim=0)
        return inputs_mix, targets_mix

    def forward(self, batch_image, batch_target, batch_group):
        return self._mixup_in_distance(batch_image, batch_target, batch_group)
