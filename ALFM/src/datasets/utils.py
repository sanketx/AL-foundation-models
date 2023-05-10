"""Utitlity classes for torchvision dataset modifications."""

import os
import os.path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import verify_str_arg

from torchvision.datasets import DatasetFolder
from torchvision.datasets import VisionDataset


class INaturalist2021(VisionDataset):
    """`iNaturalist <https://github.com/visipedia/inat_comp>`_ Dataset."""

    CATEGORIES_2021 = ["kingdom", "phylum", "class", "order", "family", "genus"]

    DATASET_URLS = {
        "2021_train": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz",
        "2021_train_mini": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz",
        "2021_valid": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz",
        # Added public test set for iNaturalist 2021 competition
        "2021_test": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.tar.gz",
    }

    DATASET_MD5 = {
        "2021_train": "e0526d53c7f7b2e3167b2b43bb2690ed",
        "2021_train_mini": "db6ed8330e634445efc8fec83ae81442",
        "2021_valid": "f6f6e0e242e3d4c9569ba56400938afc",
        # Added public test set for iNaturalist 2021 competition
        "2021_test": "7124b949fe79bfa7f7019a15ef3dbd06",
    }

    def __init__(
        self,
        root: str,
        version: str = "2021_train",
        target_type: Union[List[str], str] = "full",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.version = verify_str_arg(
            version, "version", INaturalist2021.DATASET_URLS.keys()
        )

        super().__init__(
            os.path.join(root, version),
            transform=transform,
            target_transform=target_transform,
        )

        os.makedirs(root, exist_ok=True)
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        self.all_categories: List[str] = []

        # map: category type -> name of category -> index
        self.categories_index: Dict[str, Dict[str, int]] = {}

        # list indexed by category id, containing mapping from category type -> index
        self.categories_map: List[Dict[str, int]] = []

        if not isinstance(target_type, list):
            target_type = [target_type]
        if self.version[:4] == "2021":
            self.target_type = [
                verify_str_arg(t, "target_type", ("full", *self.CATEGORIES_2021))
                for t in target_type
            ]
            self._init_2021()
        else:
            raise RuntimeError(
                f"This iNaturalist split is not supported: {self.version}"
            )

        # index of all files: (full category id, filename)
        self.index: List[Tuple[int, str]] = []

        for dir_index, dir_name in enumerate(self.all_categories):
            files = os.listdir(os.path.join(self.root, dir_name))
            for fname in files:
                self.index.append((dir_index, fname))

    def _init_2021(self) -> None:
        """Initialize based on 2021 layout"""

        self.all_categories = sorted(os.listdir(self.root))

        # map: category type -> name of category -> index
        self.categories_index = {k: {} for k in self.CATEGORIES_2021}

        for dir_index, dir_name in enumerate(self.all_categories):
            pieces = dir_name.split("_")
            if len(pieces) != 8:
                raise RuntimeError(
                    f"Unexpected category name {dir_name}, wrong number of pieces"
                )
            if pieces[0] != f"{dir_index:05d}":
                raise RuntimeError(
                    f"Unexpected category id {pieces[0]}, expecting {dir_index:05d}"
                )
            cat_map = {}
            for cat, name in zip(self.CATEGORIES_2021, pieces[1:7]):
                if name in self.categories_index[cat]:
                    cat_id = self.categories_index[cat][name]
                else:
                    cat_id = len(self.categories_index[cat])
                    self.categories_index[cat][name] = cat_id
                cat_map[cat] = cat_id
            self.categories_map.append(cat_map)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        cat_id, fname = self.index[index]
        img = Image.open(os.path.join(self.root, self.all_categories[cat_id], fname))

        target: Any = []
        for t in self.target_type:
            if t == "full":
                target.append(cat_id)
            else:
                target.append(self.categories_map[cat_id][t])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.index)

    def category_name(self, category_type: str, category_id: int) -> str:
        """
        Args:
            category_type(str): one of "full", "kingdom", "phylum", "class", "order", "family", "genus" or "super"
            category_id(int): an index (class id) from this category

        Returns:
            the name of the category
        """
        if category_type == "full":
            return self.all_categories[category_id]
        else:
            if category_type not in self.categories_index:
                raise ValueError(f"Invalid category type '{category_type}'")
            else:
                for name, id in self.categories_index[category_type].items():
                    if id == category_id:
                        return name
                raise ValueError(
                    f"Invalid category id {category_id} for {category_type}"
                )

    def _check_integrity(self) -> bool:
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0

    def download(self) -> None:
        if self._check_integrity():
            raise RuntimeError(
                f"The directory {self.root} already exists. "
                f"If you want to re-download or re-extract the images, delete the directory."
            )

        base_root = os.path.dirname(self.root)

        download_and_extract_archive(
            INaturalist2021.DATASET_URLS[self.version],
            base_root,
            filename=f"{self.version}.tgz",
            md5=INaturalist2021.DATASET_MD5[self.version],
        )

        orig_dir_name = os.path.join(
            base_root,
            os.path.basename(INaturalist2021.DATASET_URLS[self.version]).rstrip(
                ".tar.gz"
            ),
        )
        if not os.path.exists(orig_dir_name):
            raise RuntimeError(f"Unable to find downloaded files at {orig_dir_name}")
        os.rename(orig_dir_name, self.root)
        print(
            f"Dataset version '{self.version}' has been downloaded and prepared for use"
        )


class Cub2011(VisionDataset):
    base_folder = "CUB_200_2011/images"
    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self, root, train=True, transform=None, loader=default_loader, download=True
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    
class CustomImageFolder(DatasetFolder):
    def __init__(self, root, file_path, transform=None, target_transform=None, loader=default_loader):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = []

        with open(file_path, 'r') as f:
            for line in f.readlines():
                img_path, label = line.strip().split()
                self.samples.append((os.path.join(root, img_path), int(label)))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
    
