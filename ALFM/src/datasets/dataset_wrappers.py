"""Wrapper functions to standardize vision datasets."""


from typing import Optional

from torchvision import transforms
from torchvision.datasets import DTD
from torchvision.datasets import FGVCAircraft
from torchvision.datasets import Flowers102
from torchvision.datasets import Food101
from torchvision.datasets import OxfordIIITPet
from torchvision.datasets import StanfordCars
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
        return FGVCAircraft(root, split, transform, download=download)


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


class CUB200Wrapper:
    pass


class INaturalistWrapper:
    pass


class Places205Wrapper:
    pass
