import torch
from .equalized_odds import Equalized_Odds


class Strong_Pairwise_Equalized_Odds(Equalized_Odds):
    def __init__(self, interval: float = 0.02, num_group: int = 2) -> None:
        Equalized_Odds.__init__(self)
        self._tau = torch.arange(0, 1, interval)
        self._num_group = num_group

    def compute(self) -> torch.Tensor:
        target, pred, group = self._concat_states()
        differences_mean = []
        differences_max = []
        for tau in self._tau:
            defference_mean, difference_max = self._cal_difference(
                target, pred, group, tau
            )
            differences_mean.append(defference_mean)
            differences_max.append(difference_max)

        differences_mean = torch.stack(differences_mean).mean()
        differences_max = torch.stack(differences_max).mean()
        return differences_mean, differences_max
