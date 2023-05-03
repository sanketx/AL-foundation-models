"""Wrapper functions to standardize vision datasets."""


from typing import Optional

from torchvision import transforms
from torchvision.datasets import Food101
from torchvision.datasets import VisionDataset


class Food101Wrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train" if train else "test"
        return Food101(root, split, transform, download=download)
