import torch
import torch.nn as nn
from loss.binary_group import BinaryWDLoss


class MultipleWDLoss(BinaryWDLoss):
    def __init__(self):
        super().__init__()

    def _cal_multiple_wasserstein_distance(
        self,
        batch_pred: torch.Tensor,
        batch_group: torch.Tensor,
    ):
        num_group = int(batch_group.max().item() + 1)

        # group별로 split
        group_pred_list = []
        for i in range(0, num_group):
            group_pred = batch_pred[torch.where(batch_group == i)[0]]
            if group_pred.shape[0] != 0:
                group_pred_list.append(group_pred)

        # calculate wd distance
        wd_distances = []
        for i in range(0, len(group_pred_list) - 1):
            for j in range(i + 1, len(group_pred_list)):
                group_pred_i = group_pred_list[i]
                group_pred_j = group_pred_list[j]
                distance = self._cal_binary_wasserstein_distance(
                    group_pred_i,
                    group_pred_j,
                )
                wd_distances.append(distance)
        return torch.stack(wd_distances).mean()

    def forward(
        self,
        batch_pred: torch.Tensor,
        batch_group: torch.Tensor,
    ):
        wd_loss = self._cal_multiple_wasserstein_distance(
            batch_pred,
            batch_group,
        )
        return wd_loss
