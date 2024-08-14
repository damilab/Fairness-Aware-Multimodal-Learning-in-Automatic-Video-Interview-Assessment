import torch
import torch.nn as nn
from loss.binary_group import BinaryL2Loss


class MultipleL2Loss(BinaryL2Loss):
    def __init__(self):
        super().__init__()

    def _cal_multiple_l2_distance(
        self,
        batch_feature: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> torch.Tensor:
        num_group = int(batch_group.max().item() + 1)

        # group별로 split
        group_feature_list = []
        for i in range(0, num_group):
            group_feature = batch_feature[torch.where(batch_group == i)[0]]
            if group_feature.shape[0] != 0:
                group_feature_list.append(group_feature)

        # calculate l2 distance
        l2_distances = []
        for i in range(0, len(group_feature_list) - 1):
            for j in range(i + 1, len(group_feature_list)):
                distance = self._cal_binary_l2_distance(
                    group_feature_list[i],
                    group_feature_list[j],
                )
                l2_distances.append(distance)
        return torch.stack(l2_distances).mean()

    def forward(
        self,
        batch_feature: torch.Tensor,
        batch_group: torch.Tensor,
    ):
        l2_loss = self._cal_multiple_l2_distance(
            batch_feature,
            batch_group,
        )
        return l2_loss
