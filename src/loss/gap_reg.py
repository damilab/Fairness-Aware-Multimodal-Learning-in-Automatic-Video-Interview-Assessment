import torch
import torch.nn as nn


class GapReg(nn.Module):
    def __init__(self, num_group: int, mode: str):
        super().__init__()
        self._num_group = num_group
        self._mode = mode

    def _cal_gap_reg(self, batch_pred: torch.Tensor, batch_group: torch.Tensor):
        # group별로 split
        group_pred_list = []
        for i in range(self._num_group):
            group_pred = batch_pred[torch.where(batch_group == i)]
            group_pred_list.append(group_pred)

        # calculate loss gap
        loss_gap_list = []
        for i in range(len(group_pred_list) - 1):
            for j in range(i + 1, len(group_pred_list)):
                ops_0 = group_pred_list[i]
                ops_1 = group_pred_list[j]
                loss_gap = torch.abs(ops_0.mean() - ops_1.mean())
                loss_gap_list.append(loss_gap)

        if self._mode == "mean":
            return torch.stack(loss_gap_list).mean()
        elif self._mode == "max":
            return torch.stack(loss_gap_list).max()
        else:
            raise ValueError

    def forward(self, batch_pred, batch_group):
        return self._cal_gap_reg(batch_pred, batch_group)
