import torch
import torch.nn as nn
from .multiple_group import MultipleL2Loss
from .multiple_group import MultipleMMDLoss
from .multiple_group import MultipleWDLoss


class FairnessLoss(nn.Module):
    def __init__(
        self,
        config: dict,
        base_loss_function: nn.Module,
    ) -> None:
        super().__init__()
        self._base_loss_weight = config["loss"]["base_loss"]["weight"]
        self._base_loss_function = base_loss_function

        self._l2_loss_weight = config["loss"]["l2_loss"]["weight"]
        self._l2_loss_function = MultipleL2Loss()

        self._mmd_loss_weight = config["loss"]["mmd_loss"]["weight"]
        self._mmd_loss_function = MultipleMMDLoss()

        self._wd_loss_weight = config["loss"]["wd_loss"]["weight"]
        self._wd_loss_function = MultipleWDLoss()

    def forward(
        self,
        batch_pred: torch.Tensor,
        batch_feature: torch.Tensor,
        batch_target: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> dict:
        base_loss = self._base_loss_function(batch_pred, batch_target)
        l2_loss = self._l2_loss_function(batch_feature, batch_group)
        mmd_loss = self._mmd_loss_function(batch_feature, batch_group)
        wd_loss = self._wd_loss_function(batch_pred, batch_group)

        total_loss = (
            self._base_loss_weight * base_loss
            + (self._l2_loss_weight * l2_loss)
            + (self._mmd_loss_weight * mmd_loss)
            + (self._wd_loss_weight * wd_loss)
        )

        return {
            "total_loss": total_loss,
            "base_loss": base_loss,
            "l2_loss": l2_loss,
            "mmd_loss": mmd_loss,
            "wd_loss": wd_loss,
        }
