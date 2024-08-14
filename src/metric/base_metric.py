import torch
from torchmetrics import Metric


class BaseMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.add_state("_pred", default=[], dist_reduce_fx="cat")
        self.add_state("_target", default=[], dist_reduce_fx="cat")
        self.add_state("_group", default=[], dist_reduce_fx="cat")

    def _concat_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self._concat(self._pred)
        target = self._concat(self._target)
        group = self._concat(self._group)
        return target, pred, group

    def update(
        self, pred: torch.Tensor, target: torch.Tensor, group: torch.Tensor
    ) -> None:
        self._pred.append(pred.detach())
        self._target.append(target.detach())
        self._group.append(group.detach())

    def _concat(self, mat_acc: torch.Tensor) -> torch.Tensor:
        if isinstance(mat_acc, torch.Tensor):
            return mat_acc
        elif isinstance(mat_acc, list):
            return torch.cat(mat_acc)

    def _get_group_tensor(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        group: torch.Tensor,
        group_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_target = target[torch.where(group == group_idx)]
        group_pred = pred[torch.where(group == group_idx)]
        return group_target, group_pred

    def _make_binary(
        self, input_tensor: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        return torch.where(input_tensor > threshold, 1, 0).float()

    def _cal_confusion(self, target: torch.Tensor, pred: torch.Tensor) -> dict:
        target = target.bool()
        pred = pred.bool()

        tn = sum(~target * ~pred)
        fp = sum(~target * pred)
        fn = sum(target * ~pred)
        tp = sum(target * pred)

        return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    def compute(self) -> dict:
        pred = self._concat(self._pred)
        target = self._concat(self._target)
        group = self._concat(self._group)

        return {"pred": pred, "target": target, "group": group}
