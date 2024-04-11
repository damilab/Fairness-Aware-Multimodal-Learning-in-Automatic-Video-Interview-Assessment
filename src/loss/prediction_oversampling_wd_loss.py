import torch
import torch.nn as nn
import math
from functools import reduce


class PredictionOversamplingWDLoss(nn.Module):
    """batch 내에서 major group과 minor group 간 Wasserstein distance를 직접 loss로 추가"""

    def __init__(self, num_group: int, mode: str, lcm: bool):
        super().__init__()
        self._num_group = num_group
        self._mode = mode
        self._lcm = lcm

    def _cal_wasserstein_distance(
        self, batch_pred: torch.Tensor, batch_group: torch.Tensor
    ):
        # group별로 split
        group_len_list = []
        group_pred_list = []
        for i in range(self._num_group):
            group_pred = batch_pred[torch.where(batch_group == i)]
            group_len_list.append(len(group_pred))
            group_pred_list.append(group_pred)

        # prediction oversampling
        if self._lcm == True:
            max_len = reduce(lambda x, y: x * y // math.gcd(x, y), group_len_list)
        elif self._lcm == False:
            max_len = max(group_len_list)

        for idx, group_pred in enumerate(group_pred_list):
            expansion_factor = (max_len // group_pred.shape[0]) + 1
            expanded_group_pred = group_pred.repeat(expansion_factor)[:max_len]
            group_pred_list[idx] = expanded_group_pred

        # calculate wd distance
        wd_distances = []
        for i in range(len(group_pred_list) - 1):
            for j in range(i + 1, len(group_pred_list)):
                group1 = group_pred_list[i]
                group2 = group_pred_list[j]
                min_len = min(len(group1), len(group2))
                group1 = torch.sort(group1, dim=0)[0]
                group2 = torch.sort(group2, dim=0)[0]
                distance = torch.mean(torch.abs(group1 - group2))
                wd_distances.append(distance)
        if self._mode == "mean":
            return torch.stack(wd_distances).mean()
        elif self._mode == "max":
            return torch.stack(wd_distances).max()
        else:
            return 0

    def forward(self, batch_pred, batch_group):
        return self._cal_wasserstein_distance(batch_pred, batch_group)
