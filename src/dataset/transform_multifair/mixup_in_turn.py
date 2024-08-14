import torch
import torch.nn as nn
import math
from .mixup import Mixup


class MixupInTurn(nn.Module):
    def __init__(self, num_group: int):
        super().__init__()
        self._num_group = num_group
        self._num_sa = int(math.log2(num_group))
        self._group_binary_list = self._get_group_binary_list()
        self._mixup = Mixup()
        self._steps = 0

    def _get_group_binary_list(self):
        group_binary_list = []
        for i in range(self._num_group):
            group_binary_list.append(format(i, "0" + str(self._num_sa) + "b"))
        return group_binary_list

    def _mixup_in_turn(
        self,
        batch_image: torch.Tensor,
        batch_target: torch.Tensor,
        batch_group: torch.Tensor,
    ):
        iter = self._steps % self._num_sa
        self._steps += 1

        group_image_0 = []
        group_target_0 = []
        group_image_1 = []
        group_target_1 = []
        for group_binary in self._group_binary_list:
            if group_binary[iter] == "0":
                group_idx = int(group_binary, 2)
                group_image = batch_image[torch.where(batch_group == group_idx)[0]]
                group_target = batch_target[torch.where(batch_group == group_idx)[0]]
                group_image_0.append(group_image)
                group_target_0.append(group_target)
            elif group_binary[iter] == "1":
                group_idx = int(group_binary, 2)
                group_image = batch_image[torch.where(batch_group == group_idx)[0]]
                group_target = batch_target[torch.where(batch_group == group_idx)[0]]
                group_image_1.append(group_image)
                group_target_1.append(group_target)

        ####################################################
        # group 0
        group_image_0 = torch.cat(group_image_0, dim=0)
        group_target_0 = torch.cat(group_target_0, dim=0)

        indices_0 = torch.randperm(group_image_0.shape[0])
        group_image_0 = group_image_0[indices_0]
        group_target_0 = group_target_0[indices_0]
        ####################################################
        # group 1
        group_image_1 = torch.cat(group_image_1, dim=0)
        group_target_1 = torch.cat(group_target_1, dim=0)

        indices_1 = torch.randperm(group_image_1.shape[0])
        group_image_1 = group_image_1[indices_1]
        group_target_1 = group_target_1[indices_1]
        ####################################################

        mixup_results = self._mixup._mixup_once(
            image_list=[group_image_0, group_image_1],
            target_list=[group_target_0, group_target_1],
        )

        return mixup_results["inputs_mix"], mixup_results["targets_mix"]

    def forward(self, batch_image, batch_target, batch_group):
        return self._mixup_in_turn(batch_image, batch_target, batch_group)
