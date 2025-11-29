import numpy as np
from PIL import Image


def create_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """
    original — HxWx3 RGB
    mask — HxW uint8 (0/255)
    """
    orig = original.copy()
    mask_rgb = np.zeros_like(orig)

    # Красная маска
    mask_rgb[:, :, 0] = mask  # R канал
    mask_rgb[:, :, 1] = 0
    mask_rgb[:, :, 2] = 0

    overlay_np = (orig * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return Image.fromarray(overlay_np)
