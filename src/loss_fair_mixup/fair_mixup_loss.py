import torch
import torch.nn as nn
from .gap_reg import GapReg
from .mixup_loss_grad import MixupLossGrad
from .mixup_manifold_loss_grad import MixupManifoldLossGrad


class FairMixupLoss(nn.Module):
    def __init__(
        self,
        config: dict,
        base_loss_function: nn.Module,
    ) -> None:
        super().__init__()
        self._base_loss_weight = config["loss"]["base_loss"]["weight"]
        self._base_loss_function = base_loss_function

        self._gapreg_loss_weight = config["loss"]["gapreg_loss"]["weight"]
        self._gapreg_loss_function = GapReg()

        self._mixup_loss_grad_weight = config["loss"]["mixup_loss_grad"]["weight"]
        self._mixup_loss_grad_function = MixupLossGrad()

        self._mixup_manifold_loss_grad_weight = config["loss"][
            "mixup_manifold_loss_grad"
        ]["weight"]
        self._mixup_manifold_loss_grad_function = MixupManifoldLossGrad()

    def forward(
        self,
        model: nn.Module,
        batch_image: torch.Tensor,
        batch_target: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> dict:
        batch_pred, _ = model(batch_image)
        base_loss = self._base_loss_function(batch_pred, batch_target)
        if self._gapreg_loss_weight == 0:
            gapreg_loss = torch.tensor(0)
        else:
            gapreg_loss = self._gapreg_loss_function(batch_pred, batch_group)

        if self._mixup_loss_grad_weight == 0:
            mixup_loss_grad = torch.tensor(0)
        else:
            mixup_loss_grad = self._mixup_loss_grad_function(
                model, batch_image, batch_group
            )

        if self._mixup_manifold_loss_grad_weight == 0:
            mixup_manifold_loss_grad = torch.tensor(0)
        else:
            mixup_manifold_loss_grad = self._mixup_manifold_loss_grad_function(
                model, batch_image, batch_group
            )

        total_loss = (
            self._base_loss_weight * base_loss
            + self._gapreg_loss_weight * gapreg_loss
            + self._mixup_loss_grad_weight * mixup_loss_grad
            + self._mixup_manifold_loss_grad_weight * mixup_manifold_loss_grad
        )

        return {
            "total_loss": total_loss,
            "base_loss": base_loss,
            "gapreg_loss": gapreg_loss,
            "mixup_loss_grad": mixup_loss_grad,
            "mixup_manifold_loss_grad": mixup_manifold_loss_grad,
        }
