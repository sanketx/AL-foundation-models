"""Wrapper functions to standardize vision datasets."""


import os
from typing import Optional

from bcv.datasets.biology.icpr2020_pollen import ICPR2020Pollen
from bcv.datasets.cell_biology.bbbc.bbbc048_cell_cycle import BBBC048CellCycleDataset
from bcv.datasets.cell_biology.iicbu2008_hela import IICBU2008HeLa
from bcv.datasets.cytology.blood_smear.acevedo_et_al_2020 import BloodSmearDataSet
from bcv.datasets.cytology.blood_smear.malaria import MalariaDataset
from bcv.datasets.cytology.pap_smear.hussain_et_al_2019 import Hussain2019Dataset
from bcv.datasets.cytology.pap_smear.plissiti_et_al_2018 import Plissiti2018Dataset
from bcv.datasets.dermoscopy.ham10000 import HAM10000Dataset
from bcv.datasets.fundoscopy.diabetic_retinopathy import DiabeticRetinopathyDataset
from bcv.datasets.pathology.amyloid_beta.tang_et_al_2019 import AmyloidBeta2019Dataset
from bcv.datasets.pathology.idr0042_upenn_heart import UPennHeart2018Dataset
from bcv.datasets.pathology.iicbu2008_lymphoma import IICBU2008Lymphoma
from bcv.datasets.pathology.kather_et_al_2016 import ColorectalHistologyDataset
from bcv.datasets.pathology.mhist import MHist
from bcv.datasets.pathology.patch_camelyon import PatchCamelyonDataSet
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import DTD
from torchvision.datasets import SVHN
from torchvision.datasets import FGVCAircraft
from torchvision.datasets import Flowers102
from torchvision.datasets import Food101
from torchvision.datasets import ImageFolder
from torchvision.datasets import OxfordIIITPet
from torchvision.datasets import Places365
from torchvision.datasets import StanfordCars
from torchvision.datasets import VisionDataset

from ALFM.src.datasets.utils import Cub2011
from ALFM.src.datasets.utils import CustomImageFolder
from ALFM.src.datasets.utils import INaturalist2021


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
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        return Cub2011(root, train, transform=transform, download=download)


class DomainNetRealWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        root = os.path.join(root, "domainnet_real")
        file = "real_train.txt" if train else "real_test.txt"
        file = os.path.join(root, file)
        return CustomImageFolder(root, file, transform=transform)


class ImageNet100Wrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train" if train else "val"
        root = os.path.join(root, "imagenet100", split)
        return ImageFolder(root, transform=transform)


class INaturalistWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "2021_train" if train else "2021_test"
        return INaturalist2021(root, split, transform=transform, download=download)


class Places365Wrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ) -> VisionDataset:
        split = "train-standard" if train else "val"
        if download:  # Check if image archive already extracted
            try:
                Places365(
                    root, split, small=True, transform=transform, download=download
                )
            except RuntimeError:
                download = False
        return Places365(
            root, split, small=True, transform=transform, download=download
        )


class AmyloidBetaBalancedWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return AmyloidBeta2019Dataset(
            root,
            split=split,
            transform=transform,
            download=download,
            balance_classes="rand_under",
        )


class AmyloidBetaUnbalancedWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return AmyloidBeta2019Dataset(
            root,
            split=split,
            transform=transform,
            download=download,
            balance_classes=None,
        )


class BloodSmearWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return BloodSmearDataSet(
            root, split=split, transform=transform, download=download
        )


class CellCycleWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return BBBC048CellCycleDataset(
            root, split=split, transform=transform, download=download
        )


class ColonPolypsWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return MHist(root, split=split, transform=transform, download=download)


class ColorectalHistologyWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return ColorectalHistologyDataset(
            root, split=split, transform=transform, download=download
        )


class DiabeticRetinopathyWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return DiabeticRetinopathyDataset(
            root, split=split, transform=transform, download=download
        )


class HAM10000Wrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return HAM10000Dataset(
            root, split=split, transform=transform, download=download
        )


class HeartFailureWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return UPennHeart2018Dataset(
            root, split=split, transform=transform, download=download
        )


class IICBU2008HeLaWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return IICBU2008HeLa(root, split=split, transform=transform, download=download)


class IICBU2008LymphomaWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return IICBU2008Lymphoma(
            root, split=split, transform=transform, download=download
        )


class MalariaDatasetWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return MalariaDataset(root, split=split, transform=transform, download=download)


class PollenWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return ICPR2020Pollen(root, split=split, transform=transform, download=download)


class PatchCamelyonWrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train" if train else "test"
        return PatchCamelyonDataSet(
            root, split=split, transform=transform, download=download
        )


class PapSmear2018Wrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return Plissiti2018Dataset(
            root, split=split, transform=transform, download=download
        )


class PapSmear2019Wrapper:
    @staticmethod  # don't even ask
    def __call__(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = False,
    ) -> Dataset:
        split = "train+val" if train else "test"
        return Hussain2019Dataset(
            root, split=split, transform=transform, download=download
        )
