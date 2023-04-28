"""Registry of all supported model backbone."""

from enum import Enum


class ModelType(Enum):
    """Enum of supported Models."""

    openai_vit_B16 = ("ViT-B-16", "openai")

    openclip_vit_B16 = ("ViT-B-16", "laion2b_s34b_b88k")
    openclip_vit_L14 = ("ViT-L-14", "laion2b_s32b_b82k")
    openclip_vit_H14 = ("ViT-H-14", "laion2b_s32b_b79k")
    openclip_vit_g14 = ("ViT-g-14", "laion2b_s34b_b88k")
    openclip_vit_G14 = ("ViT-bigG-14", "laion2b_s39b_b160k")

    dino_vit_S14 = ("facebookresearch/dinov2", "dinov2_vits14")
    dino_vit_B14 = ("facebookresearch/dinov2", "dinov2_vitb14")
    dino_vit_L14 = ("facebookresearch/dinov2", "dinov2_vitl14")
    dino_vit_g14 = ("facebookresearch/dinov2", "dinov2_vitg14")
