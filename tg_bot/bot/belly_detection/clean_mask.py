import cv2
import numpy as np


def clean_mask(
    mask: np.ndarray,
    min_area: int = 800,
    kernel_size: int = 9,
) -> np.ndarray:
    """
    Заполняет дырки внутри маски и удаляет мелкие артефакты снаружи.
    """
    mask_bin = (mask > 0).astype(np.uint8) * 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Удаляем мелкий шум
    opened = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)

    # Заполняем дырки
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # Оставляем только крупные связные компоненты
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    clean = np.zeros_like(mask_bin)
    for i in range(1, num_labels):  # 0 — фон
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255

    return clean
