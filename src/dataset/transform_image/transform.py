from torchvision import transforms


def get_base_train_transform():
    return transforms.Compose(
        [
            transforms.Resize(size=[224, 224]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation([-20, 20]),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


def get_base_val_transform():
    return transforms.Compose(
        [
            transforms.Resize(size=[224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
