"""Registry of all supported Image Datasets."""

from enum import Enum

from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100


class DatasetType(Enum):
    """Enum of supported Datasets."""

    cifar10 = CIFAR10
    cifar100 = CIFAR100
