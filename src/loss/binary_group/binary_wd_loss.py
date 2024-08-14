import torch
import torch.nn as nn


class BinaryWDLoss(nn.Module):
    """batch 내에서 major group과 minor group 간 Wasserstein distance를 직접 loss로 추가"""

    def __init__(self):
        super().__init__()

    def _split_pred_by_group(
        self,
        batch_pred: torch.Tensor,
        batch_group: torch.Tensor,
    ):
        group_pred_0 = batch_pred[torch.where(batch_group == 0)[0]]
        group_pred_1 = batch_pred[torch.where(batch_group == 1)[0]]
        return group_pred_0, group_pred_1

    def _cal_binary_wasserstein_distance(
        self,
        group_pred_0: torch.Tensor,
        group_pred_1: torch.Tensor,
    ):
        min_len = min(len(group_pred_0), len(group_pred_1))
        group_pred_0 = torch.sort(group_pred_0, dim=0)[0][:min_len]
        group_pred_1 = torch.sort(group_pred_1, dim=0)[0][:min_len]
        return torch.mean(torch.abs(group_pred_0 - group_pred_1))

    def forward(self, batch_pred, batch_group):
        group_pred_0, group_pred_1 = self._split_pred_by_group(batch_pred, batch_group)
        return self._cal_binary_wasserstein_distance(group_pred_0, group_pred_1)
