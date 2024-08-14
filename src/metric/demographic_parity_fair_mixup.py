import torch
from .base_metric import BaseMetric


class DemographicParity_Fair_Mixup(BaseMetric):
    """statistical demographic disparity"""

    def __init__(self, num_group: int = 2) -> None:
        BaseMetric.__init__(self)
        self._num_group = num_group

    def _get_group_pred_mean(self, target, pred, group, group_idx):
        _, group_pred = self._get_group_tensor(target, pred, group, group_idx)
        return group_pred.mean()

    def _cal_difference(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        group: torch.Tensor,
    ) -> torch.Tensor:

        parities = []
        for i in range(0, self._num_group):
            parity = self._get_group_pred_mean(target, pred, group, i)
            parities.append(parity)

        disparities = []
        for i in range(0, self._num_group - 1):
            for j in range(i + 1, self._num_group):
                disparity = torch.abs(parities[i] - parities[j])
                disparities.append(disparity)
        disparities = torch.stack(disparities)
        output_mean = disparities.mean()
        output_max = disparities.max()

        return output_mean, output_max

    def compute(self) -> torch.Tensor:
        target, pred, group = self._concat_states()
        output_mean, output_max = self._cal_difference(target, pred, group)
        return output_mean, output_max
