import torch
import torch.nn as nn


class BaseVideoWeightedOverUnderSampling(nn.Module):
    def __init__(self, batch_size: int, num_group: int = 2, tau: float = 0.2):
        super().__init__()
        self._num_group = num_group
        self._batch_size = batch_size
        self._sampling_size = batch_size // num_group
        self._tau = tau

    def _cal_weights(self, batch_group_others):
        num_group_others = int(batch_group_others.max() + 1)
        n_c = batch_group_others.shape[0]
        weights_list = []
        for i in range(0, num_group_others):
            n = (batch_group_others == i).sum().item()
            weights = (n / n_c) ** self._tau
            weights_list.append(weights)

        weights_list = [weights / sum(weights_list) for weights in weights_list]
        return weights_list

    def _weighted_over_under_sampling(
        self,
        batch_video: torch.Tensor,
        batch_audio: torch.Tensor,
        batch_text: torch.Tensor,
        batch_target: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        weights_list = self._cal_weights(batch_group)
        weight_tensor = []
        for g in batch_group:
            weight_tensor.append(weights_list[g.int()])
        weight_tensor = torch.tensor(weight_tensor)

        indices = torch.multinomial(
            weight_tensor,
            batch_group.size(0),
            replacement=True,
        )

        shuffle_idx = torch.randperm(self._batch_size)

        video = batch_video[indices][shuffle_idx]
        audio = batch_audio[indices][shuffle_idx]
        text = batch_text[indices][shuffle_idx]
        target = batch_target[indices][shuffle_idx]
        group = batch_group[indices][shuffle_idx]

        return video, audio, text, target, group

    def forward(
        self,
        batch_video,
        batch_audio,
        batch_text,
        batch_target,
        batch_group,
    ):
        return self._weighted_over_under_sampling(
            batch_video,
            batch_audio,
            batch_text,
            batch_target,
            batch_group,
        )
