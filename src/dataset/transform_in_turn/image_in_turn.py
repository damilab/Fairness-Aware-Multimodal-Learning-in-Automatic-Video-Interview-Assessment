import torch
import torch.nn as nn
import math


class ImageInTurn(nn.Module):
    def __init__(self, num_group: int, device):
        super().__init__()
        self._num_group = num_group
        self._num_sa = int(math.log2(num_group))
        self._group_binary_list = self._get_group_binary_list()
        self._steps = 0
        self._device = device

    def _get_group_binary_list(self) -> list:
        group_binary_list = []
        for i in range(self._num_group):
            group_binary_list.append(format(i, "0" + str(self._num_sa) + "b"))
        return group_binary_list

    def _shuffle_tensor(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor,
        group_others: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randperm(image.shape[0])
        return image[indices], target[indices], group[indices], group_others[indices]

    def _in_turn(
        self,
        batch_image: torch.Tensor,
        batch_target: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        iter = self._steps % self._num_sa
        self._steps += 1

        group_image_0 = []
        group_target_0 = []
        group_others_0 = []

        group_image_1 = []
        group_target_1 = []
        group_others_1 = []

        for group_binary in self._group_binary_list:
            if group_binary[iter] == "0":
                group_idx = int(group_binary, 2)
                group_image = batch_image[torch.where(batch_group == group_idx)[0]]
                group_target = batch_target[torch.where(batch_group == group_idx)[0]]

                group_binary_others = group_binary[:iter] + group_binary[iter + 1 :]
                group_binary_others_idx = int(group_binary_others, 2)
                for i in range(group_image.shape[0]):
                    group_others_0.append([group_binary_others_idx])

                group_image_0.append(group_image)
                group_target_0.append(group_target)
            elif group_binary[iter] == "1":
                group_idx = int(group_binary, 2)
                group_image = batch_image[torch.where(batch_group == group_idx)[0]]
                group_target = batch_target[torch.where(batch_group == group_idx)[0]]

                group_binary_others = group_binary[:iter] + group_binary[iter + 1 :]
                group_binary_others_idx = int(group_binary_others, 2)
                for i in range(group_image.shape[0]):
                    group_others_1.append([group_binary_others_idx])

                group_image_1.append(group_image)
                group_target_1.append(group_target)

        group_image_0 = torch.cat(group_image_0, dim=0)
        group_target_0 = torch.cat(group_target_0, dim=0)
        group_label_0 = torch.zeros(group_image_0.shape[0], 1)
        group_others_0 = torch.tensor(group_others_0)

        group_image_1 = torch.cat(group_image_1, dim=0)
        group_target_1 = torch.cat(group_target_1, dim=0)
        group_label_1 = torch.ones(group_image_1.shape[0], 1)
        group_others_1 = torch.tensor(group_others_1)

        image = torch.cat([group_image_0, group_image_1], dim=0)
        target = torch.cat([group_target_0, group_target_1], dim=0)
        group = torch.cat([group_label_0, group_label_1], dim=0)
        group_others = torch.cat([group_others_0, group_others_1], dim=0)

        return image, target, group, group_others

    def forward(self, batch_image, batch_target, batch_group):
        i, t, g, go = self._in_turn(batch_image, batch_target, batch_group)
        i, t, g, go = self._shuffle_tensor(i, t, g, go)

        i = i.to(self._device)
        t = t.to(self._device)
        g = g.to(self._device)
        go = go.to(self._device)
        return i, t, g, go
