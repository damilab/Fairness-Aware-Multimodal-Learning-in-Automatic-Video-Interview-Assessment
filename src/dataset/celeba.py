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
        download: bool,
        target_attribute: str,
        sensitive_attributes: list,
    ):
        self._dataset = CelebA(
            root=root,
            split=split,
            target_type="attr",
            transform=transform,
            download=download,
        )
        self._target_attribute = target_attribute
        self._sensitive_attributes = sensitive_attributes

    def __len__(self):
        return self._dataset.__len__()

    def __getitem__(self, index):
        # image
        image, attr = self._dataset.__getitem__(index)

        # target
        target = torch.tensor([attr[celeba_attr[self._target_attribute]]])
        target = target.type(torch.float32)

        # multi
        multi = []
        for sensitive_attribute in self._sensitive_attributes:
            multi.append(attr[celeba_attr[sensitive_attribute]].item())
        multi = torch.tensor([int("".join(map(str, multi)), 2)])
        multi = multi.type(torch.float32)

        return image, target, multi
