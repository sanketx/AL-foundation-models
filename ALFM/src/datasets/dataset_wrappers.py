"""Wrapper functions to standardize vision datasets."""


from typing import Optional

from torchvision import transforms
from torchvision.datasets import DTD
from torchvision.datasets import FGVCAircraft
from torchvision.datasets import Flowers102
from torchvision.datasets import Food101
from torchvision.datasets import OxfordIIITPet
from torchvision.datasets import Places365
from torchvision.datasets import StanfordCars
from torchvision.datasets import SVHN
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


class SUN397Wrapper:
    # TO-DO: Fine-grained, 397 classes with >100 examples per class
    pass 


class StanfordCarsWrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train" if train else "test"
        return StanfordCars(root, split, transform, download=download)


class FGVCAircraftWrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "trainval" if train else "test"
        return FGVCAircraft(root, split, transform=transform, download=download)


class VOCWrapper:
    pass


class DTDWrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train" if train else "test"
        return DTD(root, split, partition=1, transform=transform, download=download)


class OxfordIIITPetWrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "trainval" if train else "test"
        return OxfordIIITPet(
            root, split, target_types="category", transform=transform, download=download
        )


class Caltech101Wrapper:
    # TO-DO: 40-800 examples per class
    pass


class Flowers102Wrapper:
    @staticmethod
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train" if train else "test"
        return Flowers102(root, split, transform, download=download)


class SVHNWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train" if train else "test"
        return SVHN(root, split, transform, download=download)


class CUB200Wrapper:
    pass


class ImageNetWrapper:
    # TO-DO: Implement subsets 100, 200 first
    pass


class INaturalistWrapper:
    # TO-DO: 2018, 2021 versions
    pass


class Places365Wrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train-standard" if train else "val"
        return Places365(root, split, small=True, transform=transform, download=download)
