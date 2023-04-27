"""Registry of all supported model backbone."""

from enum import Enum


class ModelType(Enum):
    """Enum of supported Models."""

    laion2b_vit_b16 = ("ViT-B-16", "laion2b_s34b_b88k")
    openai_vit_b16 = ("ViT-B-16", "openai")
    dino_vit_b14 = ("facebookresearch/dinov2", "dinov2_vitb14")
