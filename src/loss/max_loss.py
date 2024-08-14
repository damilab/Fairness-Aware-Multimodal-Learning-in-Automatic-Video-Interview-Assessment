import torch
import torch.nn as nn


# for single sensitive attribute
class Max_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self._base_loss = torch.nn.BCELoss()

    def _cal_max_loss(
        self,
        batch_pred: torch.Tensor,
        batch_group: torch.Tensor,
        batch_target: torch.Tensor,
    ):

        pred_0 = batch_pred[torch.where(batch_group == 0)[0]]
        pred_1 = batch_pred[torch.where(batch_group == 1)[0]]

        target_0 = batch_target[torch.where(batch_group == 0)[0]]
        target_1 = batch_target[torch.where(batch_group == 1)[0]]

        loss_0 = self._base_loss(pred_0, target_0)
        loss_1 = self._base_loss(pred_1, target_1)

        return max(loss_0, loss_1)

    def forward(self, batch_pred, batch_group, batch_target):
        return self._cal_max_loss(batch_pred, batch_group, batch_target)
