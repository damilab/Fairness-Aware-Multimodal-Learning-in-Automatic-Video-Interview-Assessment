import torch
import torch.nn as nn


class L2Loss(nn.Module):
    """Fully Connected Layer 전에 들어가는 representation 공간에서 Major group의 Z와 Minor group의 Z 간 L2 Loss를 추가"""

    """ Z - hidden space """

    def __init__(self, num_group: int, mode: str):
        super().__init__()
        self._num_group = num_group
        self._mode = mode
        self._loss_fn = nn.MSELoss()

    def _cal_l2_distance(self, batch_hidden, batch_group):
        # group별로 split
        group_hidden_list = []
        for i in range(self._num_group):
            group_hidden = batch_hidden[torch.where(batch_group == i)[0]]
            if group_hidden.shape[0] != 0:
                group_hidden_list.append(group_hidden)

        l2_distances = []
        for i in range(len(group_hidden_list) - 1):
            for j in range(i + 1, len(group_hidden_list)):
                group1 = group_hidden_list[i].mean(dim=0)
                group2 = group_hidden_list[j].mean(dim=0)
                l2_distances.append(self._loss_fn(group1, group2))

        if self._mode == "mean":
            return torch.stack(l2_distances).mean()
        elif self._mode == "max":
            return torch.stack(l2_distances).max()
        else:
            raise ValueError

    def forward(self, batch_hidden, batch_group):
        return self._cal_l2_distance(batch_hidden, batch_group)
