import torch
import torch.nn as nn
from .mixup import Mixup


class MixupViaInterpolations(nn.Module):
    def __init__(self, num_group: int):
        super().__init__()
        self._num_group = num_group
        self._mixup = Mixup()

    def _interpolations(
        self,
        image_list: list[torch.Tensor],
        target_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_list, target_list = self._mixup._slicing(image_list, target_list)
        image_interpolations = torch.stack(image_list).sum(dim=0) / len(image_list)
        target_interpolations = torch.stack(target_list).sum(dim=0) / len(target_list)
        return image_interpolations, target_interpolations

    def _mixup_via_interpolations(
        self,
        batch_image: torch.Tensor,
        batch_target: torch.Tensor,
        batch_group: torch.Tensor,
    ):
        inputs_mix_list = []
        targets_mix_list = []
        for i in range(self._num_group):
            image_origin = batch_image[torch.where(batch_group == i)[0]]
            target_origin = batch_target[torch.where(batch_group == i)[0]]
            image_others_list = []
            target_others_list = []
            for j in range(self._num_group):
                if i != j:
                    image_others = batch_image[torch.where(batch_group == j)[0]]
                    target_others = batch_target[torch.where(batch_group == j)[0]]
                    image_others_list.append(image_others)
                    target_others_list.append(target_others)

            image_interpolations, target_interpolations = self._interpolations(
                image_others_list,
                target_others_list,
            )

            mixup_results = self._mixup._mixup_once(
                image_list=[image_origin, image_interpolations],
                target_list=[target_origin, target_interpolations],
            )

            inputs_mix_list.append(mixup_results["inputs_mix"])
            targets_mix_list.append(mixup_results["targets_mix"])

        inputs_mix = torch.cat(inputs_mix_list, dim=0)
        targets_mix = torch.cat(targets_mix_list, dim=0)
        return inputs_mix, targets_mix

    def forward(self, batch_image, batch_target, batch_group):
        return self._mixup_via_interpolations(batch_image, batch_target, batch_group)
