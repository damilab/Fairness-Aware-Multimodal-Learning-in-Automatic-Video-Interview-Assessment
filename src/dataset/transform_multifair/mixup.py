import torch
import numpy as np
from numpy.random import beta


class Mixup:
    def __init__(self):
        pass

    def _slicing(
        self,
        image_list: list[torch.Tensor],
        target_list: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        image_len_list = []
        for i in range(len(image_list)):
            image_len_list.append(image_list[i].shape[0])
        min_len = min(image_len_list)

        for i in range(len(image_list)):
            image_list[i] = image_list[i][:min_len]
            target_list[i] = target_list[i][:min_len]

        return image_list, target_list

    def _mixup_once(
        self,
        image_list: list[torch.Tensor],
        target_list: list[torch.Tensor],
    ) -> dict:
        image_list, target_list = self._slicing(
            image_list,
            target_list,
        )
        num_group = len(image_list)
        gamma_list = np.random.dirichlet(np.ones(num_group), size=1).tolist()[0]
        inputs_mix = image_list[0] * gamma_list[0]
        targets_mix = target_list[0] * gamma_list[0]
        for i in range(1, num_group):
            inputs_mix += image_list[i] * gamma_list[i]
            targets_mix += target_list[i] * gamma_list[i]

        return {
            "inputs_mix": inputs_mix,
            "targets_mix": targets_mix,
        }
