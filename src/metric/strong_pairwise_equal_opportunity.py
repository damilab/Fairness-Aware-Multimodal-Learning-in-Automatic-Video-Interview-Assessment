import torch
from .epual_opportunity import Equal_Opportunity


class Strong_Pairwise_Equal_Opportunity(Equal_Opportunity):
    def __init__(self, interval: float = 0.02, num_group: int = 2) -> None:
        Equal_Opportunity.__init__(self)
        self._tau = torch.arange(0, 1, interval)
        self._num_group = num_group

    def _get_difference(
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
        return difference

    def compute(self) -> torch.Tensor:
        target, pred, group = self._concat_states()

        differences = []
        for tau in self._tau:
            difference = self._get_difference(target, pred, group, tau)
            differences.append(difference)

        differences = torch.stack(differences)
        differences_mean = differences.mean()
        differences_max = differences.max(dim=0).values.mean()
        return differences_mean, differences_max
