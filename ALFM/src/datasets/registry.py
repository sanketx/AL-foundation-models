"""Registry of all supported Image Datasets."""

from enum import Enum

from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

from ALFM.src.datasets.dataset_wrappers import DTDWrapper
from ALFM.src.datasets.dataset_wrappers import FGVCAircraftWrapper
from ALFM.src.datasets.dataset_wrappers import Flowers102Wrapper
from ALFM.src.datasets.dataset_wrappers import Food101Wrapper
from ALFM.src.datasets.dataset_wrappers import OxfordIIITPetWrapper
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
