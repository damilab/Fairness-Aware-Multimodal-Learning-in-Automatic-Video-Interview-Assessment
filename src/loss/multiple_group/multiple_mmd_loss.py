import torch
from loss.binary_group import BinaryMMDLoss


class MultipleMMDLoss(BinaryMMDLoss):
    def __init__(self):
        super().__init__()

    def _cal_multiple_maximum_mean_discrepancy(
        self,
        batch_feature: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> torch.Tensor:
        num_group = int(batch_group.max().item() + 1)

        # group별로 split
        group_feature_list = []
        for i in range(0, num_group):
            group_feature = batch_feature[torch.where(batch_group == i)[0]]
            if group_feature.shape[0] != 0:
                group_feature_list.append(group_feature)

        # calculate mmd distance
        maximum_mean_discrepancy_list = []
        for i in range(0, len(group_feature_list) - 1):
            for j in range(i + 1, len(group_feature_list)):
                maximum_mean_discrepancy = self._cal_binary_maximum_mean_discrepancy(
                    group_feature_list[i],
                    group_feature_list[j],
                )
                maximum_mean_discrepancy_list.append(maximum_mean_discrepancy)
        return torch.stack(maximum_mean_discrepancy_list).mean()

    def forward(
        self,
        batch_feature: torch.Tensor,
        batch_group: torch.Tensor,
    ):
        mmd_loss = self._cal_multiple_maximum_mean_discrepancy(
            batch_feature,
            batch_group,
        )
        return mmd_loss
