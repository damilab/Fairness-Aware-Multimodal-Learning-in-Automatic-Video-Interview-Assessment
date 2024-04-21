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

        # prediction oversampling (max_len or lcm)
        if self._lcm == True:
            max_len = reduce(lambda x, y: x * y // math.gcd(x, y), group_len_list)
        elif self._lcm == False:
            max_len = max(group_len_list)

        expanded_group_pred_list = []
        for idx, group_pred in enumerate(group_pred_list):
            try:  # batch 내에서 특정 그룹의 sample이 없다면 pass하고 나머지 그룹에 대해서 wasserstein distance 구하기
                expansion_factor = (max_len // group_pred.shape[0]) + 1
                expanded_group_pred = group_pred.repeat(expansion_factor)[:max_len]
                expanded_group_pred_list.append(expanded_group_pred)
            except:
                pass

        # calculate wd distance
        wd_distances = []
        for i in range(len(expanded_group_pred_list) - 1):
            for j in range(i + 1, len(expanded_group_pred_list)):
                group1 = expanded_group_pred_list[i]
                group2 = expanded_group_pred_list[j]
                # min_len = min(len(group1), len(group2))
                group1 = torch.sort(group1, dim=0)[0]
                group2 = torch.sort(group2, dim=0)[0]
                distance = torch.mean(torch.abs(group1 - group2))
                wd_distances.append(distance)
        if self._mode == "mean":
            return torch.stack(wd_distances).mean()
        elif self._mode == "max":
            return torch.stack(wd_distances).max()
        else:
            raise ValueError

    def forward(self, batch_pred, batch_group):
        return self._cal_wasserstein_distance(batch_pred, batch_group)
