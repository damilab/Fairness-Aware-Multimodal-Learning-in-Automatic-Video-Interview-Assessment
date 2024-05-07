import torch
import torch.nn as nn


class FairnessLoss(nn.Module):
    def __init__(
        self,
        base_loss_weight: float,
        base_loss_function: nn.Module,
        wd_loss_weight: float,
        wd_loss_function: nn.Module,
        powd_loss_weight: float,
        powd_loss_function: nn.Module,
        gapreg_loss_weight: float,
        gapreg_loss_function: nn.Module,
        l2_loss_weight: float,
        l2_loss_function: nn.Module,
    ) -> dict:
        super().__init__()

        self._base_loss_weight = base_loss_weight
        self._base_loss_function = base_loss_function

        self._l2_loss_weight = l2_loss_weight
        self._l2_loss_function = l2_loss_function

        self._wd_loss_weight = wd_loss_weight
        self._wd_loss_function = wd_loss_function

        self._powd_loss_weight = powd_loss_weight
        self._powd_loss_function = powd_loss_function

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
        l2_loss = self._l2_loss_function(feature, group)
        wd_loss = self._wd_loss_function(pred, group)
        powd_loss = self._powd_loss_function(pred, group)
        gapreg_loss = self._gapreg_loss_function(pred, group)

        total_loss = (
            self._base_loss_weight * base_loss
            + (self._l2_loss_weight * l2_loss)
            + (self._wd_loss_weight * wd_loss)
            + (self._powd_loss_weight * powd_loss)
            + (self._gapreg_loss_weight * gapreg_loss)
        )

        return {
            "total_loss": total_loss,
            "base_loss": base_loss,
            "l2_loss": l2_loss,
            "wd_loss": wd_loss,
            "powd_loss": powd_loss,
            "gapreg_loss": gapreg_loss,
        }
