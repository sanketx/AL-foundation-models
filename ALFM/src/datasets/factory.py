"""Dataset Factory which creates Datasets from Enums."""

from typing import Optional

from ALFM.src.datasets.registry import DatasetType
from torchvision import transforms
from torchvision.datasets import VisionDataset


def create_dataset(
    dataset_type: DatasetType,
    root: str,
    train: bool,
    transform: Optional[transforms.Compose] = None,
) -> VisionDataset:
    """Create a dataset given its corresponding DatasetType enum value.

    Args:
        dataset_type (DatasetType): An enum value representing the dataset to be created.
        root (str): The root directory where the dataset is stored or should be downloaded.
        train (bool): If True, the dataset represents the training set; otherwise, it's the test set.
        transform (Optional[Callable[[Image.Image], torch.Tensor]]):
            An optional transform function to be applied to the dataset images.
            It takes a PIL image as input and returns a PyTorch tensor as output.

    Returns:
        VisionDataset: An instance of the dataset specified by the DatasetType enum value.

    """
    return dataset_type.value(root, train=train, transform=transform)
