"""Registry of all supported Image Datasets."""

from enum import Enum

from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

from ALFM.src.datasets.dataset_wrappers import AmyloidBetaBalancedWrapper
from ALFM.src.datasets.dataset_wrappers import AmyloidBetaUnbalancedWrapper
from ALFM.src.datasets.dataset_wrappers import BloodSmearWrapper
from ALFM.src.datasets.dataset_wrappers import CellCycleWrapper
from ALFM.src.datasets.dataset_wrappers import ColonPolypsWrapper
from ALFM.src.datasets.dataset_wrappers import ColorectalHistologyWrapper
from ALFM.src.datasets.dataset_wrappers import CUB200Wrapper
from ALFM.src.datasets.dataset_wrappers import DiabeticRetinopathyWrapper
from ALFM.src.datasets.dataset_wrappers import DomainNetRealWrapper
from ALFM.src.datasets.dataset_wrappers import DTDWrapper
from ALFM.src.datasets.dataset_wrappers import FGVCAircraftWrapper
from ALFM.src.datasets.dataset_wrappers import Flowers102Wrapper
from ALFM.src.datasets.dataset_wrappers import Food101Wrapper
from ALFM.src.datasets.dataset_wrappers import HAM10000Wrapper
from ALFM.src.datasets.dataset_wrappers import HeartFailureWrapper
from ALFM.src.datasets.dataset_wrappers import IICBU2008HeLaWrapper
from ALFM.src.datasets.dataset_wrappers import IICBU2008LymphomaWrapper
from ALFM.src.datasets.dataset_wrappers import ImageNet100Wrapper
from ALFM.src.datasets.dataset_wrappers import INaturalistWrapper
from ALFM.src.datasets.dataset_wrappers import MalariaDatasetWrapper
from ALFM.src.datasets.dataset_wrappers import OxfordIIITPetWrapper
from ALFM.src.datasets.dataset_wrappers import PapSmear2018Wrapper
from ALFM.src.datasets.dataset_wrappers import PapSmear2019Wrapper
from ALFM.src.datasets.dataset_wrappers import PatchCamelyonWrapper
from ALFM.src.datasets.dataset_wrappers import Places365Wrapper
from ALFM.src.datasets.dataset_wrappers import PollenWrapper
from ALFM.src.datasets.dataset_wrappers import StanfordCarsWrapper
from ALFM.src.datasets.dataset_wrappers import SVHNWrapper


class DatasetType(Enum):
    """Enum of supported Datasets."""

    cifar10 = CIFAR10
    cifar100 = CIFAR100
    food101 = Food101Wrapper()
    cars = StanfordCarsWrapper()
    aircraft = FGVCAircraftWrapper()
    dtd = DTDWrapper()
    pets = OxfordIIITPetWrapper()
    flowers = Flowers102Wrapper()
    svhn = SVHNWrapper()
    places365 = Places365Wrapper()
    cub200 = CUB200Wrapper()
    inat2021 = INaturalistWrapper()
    imagenet100 = ImageNet100Wrapper()
    domainnetreal = DomainNetRealWrapper()
    amyloid_beta_bal = AmyloidBetaBalancedWrapper()
    amyloid_beta_unbal = AmyloidBetaUnbalancedWrapper()
    blood_smear = BloodSmearWrapper()
    cell_cycle = CellCycleWrapper()
    colon_polyps = ColonPolypsWrapper()
    colorectal_histology = ColorectalHistologyWrapper()
    diabetic_retinopathy = DiabeticRetinopathyWrapper()
    ham10000 = HAM10000Wrapper()
    heart_failure = HeartFailureWrapper()
    iicbu_hela = IICBU2008HeLaWrapper()
    iicbu_lymphoma = IICBU2008LymphomaWrapper()
    malaria = MalariaDatasetWrapper()
    pap_smear_2018 = PapSmear2018Wrapper()
    pap_smear_2019 = PapSmear2019Wrapper()
    patch_camelyon = PatchCamelyonWrapper()
    pollen = PollenWrapper()
