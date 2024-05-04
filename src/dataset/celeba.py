import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from dataset.celeba_attr import celeba_attr


class CelebADataset_MSA(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: nn.Module,
        gaussian_noise_transform: bool,
        download: bool,
        target_attribute: str,
        sensitive_attributes: list,
    ):
        self._gaussian_noise_transform = gaussian_noise_transform
        self._dataset = CelebA(
            root=root,
            split=split,
            target_type="attr",
            transform=transform,
            download=download,
        )
        self._target_attribute = target_attribute
        self._sensitive_attributes = sensitive_attributes

    def gaussian_noise_transform(self, image: torch.tensor):
        std_dev = 0.1  # 표준 편차 설정
        mean = 0.0  # 평균 설정
        noise = torch.randn_like(image) * std_dev + mean
        image = image + noise
        return image

    def __len__(self):
        return self._dataset.__len__()

    def __getitem__(self, index):
        image, attr = self._dataset.__getitem__(index)
        target = torch.tensor([attr[celeba_attr[self._target_attribute]]])

        multi = []
        for sensitive_attribute in self._sensitive_attributes:
            multi.append(attr[celeba_attr[sensitive_attribute]].item())
        multi = torch.tensor([int("".join(map(str, multi)), 2)])
        # Gaussian noise
        if self._gaussian_noise_transform == True:
            image = self.gaussian_noise_transform(image)

        return image, target, multi
