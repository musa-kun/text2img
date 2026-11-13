"""Dataset and DataLoader utilities for text-conditioned Fashion-MNIST."""
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

CLASS_NAMES = [
    "t-shirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot",
]

CAPTION_TEMPLATES = [
    "a photo of a {}",
    "an image of a {}",
    "a small picture of a {}",
    "a grayscale {}",
    "{}",
]


class CaptionedFashionMNIST(Dataset):
    """Fashion-MNIST images paired with template-generated text captions.

    Returns:
        image:   FloatTensor of shape (1, image_size, image_size), values in [-1, 1]
        caption: str, e.g. "a photo of a sneaker"
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        image_size: int = 32,
        random_flip: bool = True,
        download: bool = True,
    ):
        tfms = [transforms.Resize(image_size)]
        if random_flip and train:
            tfms.append(transforms.RandomHorizontalFlip())
        tfms += [
            transforms.ToTensor(),                      # [0, 1]
            transforms.Normalize((0.5,), (0.5,)),       # -> [-1, 1]
        ]
        self.base = datasets.FashionMNIST(
            root=root, train=train, download=download,
            transform=transforms.Compose(tfms),
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image, label = self.base[idx]
        template = random.choice(CAPTION_TEMPLATES)
        caption = template.format(CLASS_NAMES[label])
        return image, caption


def get_dataloader(
    root: str = "./data",
    train: bool = True,
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 2,
) -> DataLoader:
    dataset = CaptionedFashionMNIST(root=root, train=train, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )
