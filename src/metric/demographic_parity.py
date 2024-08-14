import torch
from .base_metric import BaseMetric


class DemographicParity(BaseMetric):
    """statistical demographic disparity"""

    def __init__(self, tau: float = 0.5, num_group: int = 2) -> None:
        BaseMetric.__init__(self)
        self._tau = tau
        self._num_group = num_group

    def _get_group_parity(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        group: torch.Tensor,
        group_idx: int,
    ):
        group_target, group_pred = self._get_group_tensor(
            target, pred, group, group_idx
        )
        group_confusion = self._cal_confusion(group_target, group_pred)
        # RPP : Rate of Positive Predictions = FP + TP / FN + FP + TN + TP
        tot_data = (
            group_confusion["fp"]
            + group_confusion["tp"]
            + group_confusion["fn"]
            + group_confusion["tn"]
        )
        group_parity = (group_confusion["fp"] + group_confusion["tp"]) / (
            tot_data + 1e-10
        )
        return group_parity

    def _cal_difference(
        self, target: torch.Tensor, pred: torch.Tensor, group: torch.Tensor, tau: float
    ) -> torch.Tensor:
        target_binary = self._make_binary(target, tau)
        pred_binary = self._make_binary(pred, tau)

        parities = []
        for i in range(0, self._num_group):
            parity = self._get_group_parity(target_binary, pred_binary, group, i)
            parities.append(parity)

        difference = []
        for i in range(0, self._num_group - 1):
            for j in range(i + 1, self._num_group):
                disparity = torch.abs(parities[i] - parities[j])
                difference.append(disparity)
        difference = torch.stack(difference)

        output_mean = difference.mean()
        output_max = difference.max()
        return output_mean, output_max

    def compute(self) -> torch.Tensor:
        target, pred, group = self._concat_states()
        output_mean, output_max = self._cal_difference(target, pred, group, self._tau)
        return output_mean, output_max
