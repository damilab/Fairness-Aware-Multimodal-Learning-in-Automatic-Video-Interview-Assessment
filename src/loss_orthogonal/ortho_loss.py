import torch
import torch.nn as nn
from .row_ortho_loss import RowOrthLoss
from .col_ortho_loss import ColOrthLoss


class OrthoLoss(nn.Module):
    def __init__(
        self,
        config: dict,
        base_loss_function: nn.Module,
        U_list: list,
    ) -> None:
        super().__init__()
        self._base_loss_weight = config["loss"]["base_loss"]["weight"]
        self._base_loss_function = base_loss_function

        device = config["trainer"]["device"]
        self.tau = config["metric"]["tau"]

        self._col_conditional = config["loss"]["col_loss"]["conditional"]
        self._col_loss_weight = config["loss"]["col_loss"]["weight"]
        self._col_loss_fn = ColOrthLoss(
            U_list=U_list,
            margin=config["loss"]["col_loss"]["margin"],
            threshold=config["loss"]["col_loss"]["threshold"],
            moving_base=config["loss"]["col_loss"]["moving_base"],
            moving_epoch=config["loss"]["col_loss"]["moving_epoch"],
            device=device,
        )

        self._row_conditional = config["loss"]["row_loss"]["conditional"]
        self._row_loss_weight = config["loss"]["row_loss"]["weight"]
        self._row_loss_fn = RowOrthLoss(
            margin=config["loss"]["row_loss"]["margin"],
        )

    def _split(
        self,
        feature_t: torch.Tensor,
        feature_a: torch.Tensor,
        target: torch.Tensor,
        conditional: bool,
    ) -> tuple[list, list]:
        if conditional:
            indices0 = torch.where(target < self.tau)[0]
            indices1 = torch.where(target >= self.tau)[0]

            feature_t_0 = torch.index_select(feature_t, 0, indices0)
            feature_t_1 = torch.index_select(feature_t, 0, indices1)

            feature_a_0 = torch.index_select(feature_a, 0, indices0)
            feature_a_1 = torch.index_select(feature_a, 0, indices1)

            t_list = [feature_t_0, feature_t_1]
            a_list = [feature_a_0, feature_a_1]
        else:
            t_list = [feature_t]
            a_list = [feature_a]

        return t_list, a_list

    def forward(
        self,
        batch_pred: torch.Tensor,
        batch_feature_t: torch.Tensor,
        batch_feature_a: torch.Tensor,
        batch_target: torch.Tensor,
        epoch: int,
    ) -> dict:
        base_loss = self._base_loss_function(batch_pred, batch_target)

        t_list, a_list = self._split(
            batch_feature_t,
            batch_feature_a,
            batch_target,
            self._col_conditional,
        )

        col_loss = self._col_loss_fn(
            t_list,
            a_list,
            epoch,
        )

        t_list, a_list = self._split(
            batch_feature_t,
            batch_feature_a,
            batch_target,
            self._row_conditional,
        )

        row_loss = self._row_loss_fn(
            t_list,
            a_list,
        )

        total_loss = (
            self._base_loss_weight * base_loss
            + (self._col_loss_weight * col_loss)
            + (self._row_loss_weight * row_loss)
        )

        return {
            "total_loss": total_loss,
            "base_loss": base_loss,
            "col_loss": col_loss,
            "row_loss": row_loss,
        }
