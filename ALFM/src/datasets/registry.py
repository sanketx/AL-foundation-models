"""Registry of all supported Image Datasets."""

from enum import Enum

from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

from ALFM.src.datasets.dataset_wrappers import AmyloidBetaWrapper
from ALFM.src.datasets.dataset_wrappers import BloodSmearWrapper
from ALFM.src.datasets.dataset_wrappers import CellCycleWrapper
from ALFM.src.datasets.dataset_wrappers import ColonPolypsWrapper
from ALFM.src.datasets.dataset_wrappers import ColorectalHistologyWrapper
from ALFM.src.datasets.dataset_wrappers import DTDWrapper
from ALFM.src.datasets.dataset_wrappers import FGVCAircraftWrapper
from ALFM.src.datasets.dataset_wrappers import Flowers102Wrapper
from ALFM.src.datasets.dataset_wrappers import Food101Wrapper
from ALFM.src.datasets.dataset_wrappers import HeartFailureWrapper
from ALFM.src.datasets.dataset_wrappers import LiquidBasedCytologyWrapper
from ALFM.src.datasets.dataset_wrappers import OxfordIIITPetWrapper
from ALFM.src.datasets.dataset_wrappers import PatchCamelyonWrapper
from ALFM.src.datasets.dataset_wrappers import Places365Wrapper
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
    amyloid_beta = AmyloidBetaWrapper()
    blood_smear = BloodSmearWrapper()
    cell_cycle = CellCycleWrapper()
    colon_polyps = ColonPolypsWrapper()
    colorectal_histology = ColorectalHistologyWrapper()
    heart_failure = HeartFailureWrapper()
    liquid_based_cytology = LiquidBasedCytologyWrapper()
    patch_camelyon = PatchCamelyonWrapper()
