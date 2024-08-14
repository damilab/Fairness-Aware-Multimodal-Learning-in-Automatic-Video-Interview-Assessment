import torch
import torch.nn as nn


class GapReg(nn.Module):
    def __init__(self):
        super().__init__()

    def _cal_gap_reg(
        self,
        batch_pred: torch.Tensor,
        batch_group: torch.Tensor,
    ):
        num_group = int(batch_group.max().item() + 1)

        # group별로 split
        group_pred_list = []
        for i in range(num_group):
            group_pred = batch_pred[torch.where(batch_group == i)]
            if group_pred.shape[0] != 0:
                group_pred_list.append(group_pred)

        # calculate loss gap
        loss_gap_list = []
        for i in range(len(group_pred_list) - 1):
            for j in range(i + 1, len(group_pred_list)):
                ops_0 = group_pred_list[i]
                ops_1 = group_pred_list[j]
                loss_gap = torch.abs(ops_0.mean() - ops_1.mean())
                loss_gap_list.append(loss_gap)

        return torch.stack(loss_gap_list).mean()

    def forward(self, batch_pred, batch_group):
        return self._cal_gap_reg(batch_pred, batch_group)
