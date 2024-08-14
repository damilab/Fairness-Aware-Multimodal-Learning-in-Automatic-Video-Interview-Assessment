from .transform_image import get_base_train_transform
from .transform_image import get_base_val_transform
from .celeba import CelebADataset_MSA
from torch.utils.data import DataLoader


def get_celeba_dataloader(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = get_base_train_transform()
    valid_transform = get_base_val_transform()

    # dataset
    train_dataset = CelebADataset_MSA(
        root=config["datamodule"]["image_dir"],
        split="train",
        transform=train_transform,
        download=False,
        target_attribute=config["datamodule"]["target_attribute"],
        sensitive_attributes=config["datamodule"]["sensitive_attributes"],
    )
    valid_dataset = CelebADataset_MSA(
        root=config["datamodule"]["image_dir"],
        split="valid",
        transform=valid_transform,
        download=False,
        target_attribute=config["datamodule"]["target_attribute"],
        sensitive_attributes=config["datamodule"]["sensitive_attributes"],
    )
    test_dataset = CelebADataset_MSA(
        root=config["datamodule"]["image_dir"],
        split="test",
        transform=valid_transform,
        download=False,
        target_attribute=config["datamodule"]["target_attribute"],
        sensitive_attributes=config["datamodule"]["sensitive_attributes"],
    )

    # dataloader
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        num_workers=config["datamodule"]["num_workers"],
        batch_size=config["datamodule"]["batch_size"],
        persistent_workers=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        drop_last=False,
        num_workers=config["datamodule"]["num_workers"],
        batch_size=config["datamodule"]["batch_size"],
        persistent_workers=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        num_workers=config["datamodule"]["num_workers"],
        batch_size=config["datamodule"]["batch_size"],
        persistent_workers=True,
    )
    return train_dataloader, valid_dataloader, test_dataloader
