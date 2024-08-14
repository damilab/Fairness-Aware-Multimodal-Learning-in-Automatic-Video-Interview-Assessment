import torch
from .base_metric import BaseMetric


class Equal_Opportunity(BaseMetric):
    def __init__(self, tau: float = 0.5, num_group: int = 2) -> None:
        BaseMetric.__init__(self)
        self._tau = tau
        self._num_group = num_group

    def _get_group_score(
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
        # TPR : True Positive Rate = TP / TP + FN
        true_positive_rate = group_confusion["tp"] / (
            group_confusion["tp"] + group_confusion["fn"] + 1e-10
        )
        return true_positive_rate

    def _cal_difference(
        self, target: torch.Tensor, pred: torch.Tensor, group: torch.Tensor, tau: float
    ) -> torch.Tensor:
        target_binary = self._make_binary(target, tau)
        pred_binary = self._make_binary(pred, tau)

        parities = []
        for i in range(0, self._num_group):
            parity = self._get_group_score(target_binary, pred_binary, group, i)
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
        difference = self._cal_difference(target, pred, group, self._tau)

        return difference
