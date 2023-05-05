"""Model Factory which creates Models from Enums."""

from typing import Tuple
from typing import cast

import open_clip
import torch
from torch import nn
from torchvision import transforms

from ALFM.src.models.registry import ModelType


def dino_transform(resize_size: int = 256, crop_size: int = 224) -> transforms.Compose:
    """Returns the torchvision Transforms for dino v2 models.

    Args:
    resize_size (int, optional): The size of the smallest edge of the image after resizing. Default is 256.
    crop_size (int, optional): The size of the image after center cropping. Default is 224.

    Returns:
    transforms.Compose: A composition of transformations to be applied to the input image.
    """
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    return transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def create_model(
    model_type: ModelType,
    cache_dir: str,
) -> Tuple[nn.Module, transforms.Compose]:
    """Create a model and its associated transformation given a model type and cache directory.

    Args:
        model_type (ModelType): The type of model to create, specified as an enum value.
        cache_dir (str): The directory where cached/pretrained models are stored.

    Returns:
        Tuple[nn.Module, transforms.Compose]: A tuple containing the instantiated model
            (a subclass of nn.Module) and the associated transformation (transforms.Compose).
    """
    match model_type:
        case ModelType.openclip_vit_B16 | ModelType.openclip_vit_L14 | ModelType.openclip_vit_H14 | ModelType.openclip_vit_g14 | ModelType.openclip_vit_G14 | ModelType.openai_vit_B16:
            result = open_clip.create_model_from_pretrained(
                *model_type.value, cache_dir=cache_dir
            )
            result = cast(Tuple[open_clip.CLIP, transforms.Compose], result)
            model, transform = result

            model = model.visual

        case ModelType.dino_vit_S14 | ModelType.dino_vit_B14 | ModelType.dino_vit_L14 | ModelType.dino_vit_g14:
            torch.hub.set_dir(cache_dir)  # type: ignore[no-untyped-call]
            model = torch.hub.load(*model_type.value)  # type: ignore[no-untyped-call]
            model = cast(nn.Module, model)
            transform = dino_transform()

    return model, transform
