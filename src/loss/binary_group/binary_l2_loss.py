import torch
import torch.nn as nn


class BinaryL2Loss(nn.Module):
    """Fully Connected Layer 전에 들어가는 representation 공간에서 Major group의 Z와 Minor group의 Z 간 L2 Loss를 추가"""

    """ Z - hidden space """

    def __init__(self):
        super().__init__()
        self._loss_fn = nn.MSELoss()

    def _split_feature_by_group(
        self,
        batch_feature: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_feature_0 = batch_feature[torch.where(batch_group == 0)[0]]
        group_feature_1 = batch_feature[torch.where(batch_group == 1)[0]]
        return group_feature_0, group_feature_1

    def _cal_binary_l2_distance(
        self,
        group_feature_0: torch.Tensor,
        group_feature_1: torch.Tensor,
    ) -> torch.Tensor:
        return self._loss_fn(group_feature_0.mean(dim=0), group_feature_1.mean(dim=0))

    def forward(
        self,
        batch_feature: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> torch.Tensor:
        group_feature_0, group_feature_1 = self._split_feature_by_group(
            batch_feature,
            batch_group,
        )
        l2_loss = self._cal_binary_l2_distance(
            group_feature_0,
            group_feature_1,
        )
        return l2_loss
