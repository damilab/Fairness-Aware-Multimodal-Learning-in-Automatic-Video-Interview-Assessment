import torch
import torch.nn as nn


class ImageWeightedOverUnderSampling(nn.Module):
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
        batch_image: torch.Tensor,
        batch_target: torch.Tensor,
        batch_group: torch.Tensor,
        batch_group_others: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        group_image_list = []
        group_target_list = []
        group_label_list = []

        for i in range(0, self._num_group):
            group_others = batch_group_others[torch.where(batch_group == i)[0]]
            weights_list = self._cal_weights(group_others)
            weight_tensor = []
            for g in group_others:
                weight_tensor.append(weights_list[g])
            weight_tensor = torch.tensor(weight_tensor)
            indices = torch.multinomial(
                weight_tensor,
                self._sampling_size,
                replacement=True,
            )

            group_image = batch_image[torch.where(batch_group == i)[0]][indices]
            target = batch_target[torch.where(batch_group == i)[0]][indices]
            group = batch_group[torch.where(batch_group == i)[0]][indices]

            group_image_list.append(group_image)
            group_target_list.append(target)
            group_label_list.append(group)

        # shuffle
        indices = torch.randperm(self._batch_size)
        image = torch.cat(group_image_list, dim=0)[indices]
        target = torch.cat(group_target_list, dim=0)[indices]
        group = torch.cat(group_label_list, dim=0)[indices]

        return image, target, group

    def forward(
        self,
        batch_image,
        batch_target,
        batch_group,
        batch_group_others,
    ):
        return self._weighted_over_under_sampling(
            batch_image,
            batch_target,
            batch_group,
            batch_group_others,
        )
