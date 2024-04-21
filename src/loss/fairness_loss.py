import torch
import torch.nn as nn
from .gap_reg import GapReg


class FairnessLoss(nn.Module):
    def __init__(
        self,
        base_loss_weight: float,
        base_loss_function: nn.Module,
        wd_loss_weight: float,
        wd_loss_function: nn.Module,
        gapreg_loss_weight: float,
        gapreg_loss_function: nn.Module,
    ) -> dict:
        super().__init__()

        self._base_loss_weight = base_loss_weight
        self._base_loss_function = base_loss_function

        self._wd_loss_weight = wd_loss_weight
        self._wd_loss_function = wd_loss_function

        self._gapreg_loss_weight = gapreg_loss_weight
        self._gapreg_loss_function = gapreg_loss_function

    def forward(
        self,
        feature: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor,
    ) -> dict:
        base_loss = self._base_loss_function(pred, target)
        wd_loss = self._wd_loss_function(pred, group)
        gapreg_loss = self._gapreg_loss_function(pred, group)

        total_loss = (
            (self._base_loss_weight * base_loss)
            + (self._wd_loss_weight * wd_loss)
            + (self._gapreg_loss_weight * gapreg_loss)
        )

        return {
            "total_loss": total_loss,
            "base_loss": base_loss,
            "wd_loss": wd_loss,
            "gapreg_loss": gapreg_loss,
        }
