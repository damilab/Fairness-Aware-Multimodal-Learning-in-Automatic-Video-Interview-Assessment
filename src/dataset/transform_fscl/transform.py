from torchvision import transforms


class TwoCropTransform:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_fscl_train_transform():
    return TwoCropTransform()
