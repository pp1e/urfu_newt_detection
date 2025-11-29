import numpy as np
import cv2
from PIL import Image


def warp_belly_to_rect(image_np: np.ndarray, target_size: tuple[int, int]) -> Image.Image | None:
    """Warp the belly along its medial axis so that it fills the entire rectangle."""
    mask_bool = image_np.sum(axis=2) > 0
    if not mask_bool.any():
        return None

    # Align major axis vertically using PCA on mask coordinates.
    ys, xs = np.nonzero(mask_bool)
    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1]  # eigenvector with largest eigenvalue
    angle_rad = np.arctan2(major[1], major[0])
    rot_deg = 90.0 - np.degrees(angle_rad)

    h, w = mask_bool.shape
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
    rot_image = cv2.warpAffine(image_np, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rot_mask = cv2.warpAffine(
        (mask_bool.astype(np.uint8) * 255), rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
    ) > 0

    # Recompute row spans on rotated mask.
    x_min = np.full(h, np.inf)
    x_max = np.full(h, -np.inf)
    ys_nonzero, xs_nonzero = np.nonzero(rot_mask)
    if len(ys_nonzero) == 0:
        return None
    for y, x in zip(ys_nonzero, xs_nonzero):
        if x < x_min[y]:
            x_min[y] = x
        if x > x_max[y]:
            x_max[y] = x

    valid_rows = np.where(x_max >= 0)[0]
    if len(valid_rows) == 0:
        return None

    y_start, y_end = valid_rows.min(), valid_rows.max()
    tgt_w, tgt_h = target_size
    out = np.zeros((tgt_h, tgt_w, 3), dtype=np.uint8)

    src_y_f = np.linspace(y_start, y_end, tgt_h)
    for yi, sy in enumerate(src_y_f):
        sy_idx = int(round(sy))
        # Find nearest valid row if current is empty.
        if x_max[sy_idx] < 0:
            nearest = valid_rows[np.abs(valid_rows - sy_idx).argmin()]
            sy_idx = int(nearest)
        row = rot_image[sy_idx]
        x0, x1 = int(x_min[sy_idx]), int(x_max[sy_idx])
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w - 1))
        if x1 <= x0:
            continue
        xs_src = np.linspace(x0, x1, tgt_w)
        # Interpolate per channel.
        for c in range(3):
            out[yi, :, c] = np.interp(xs_src, np.arange(w), row[:, c]).astype(np.uint8)

    return Image.fromarray(out)

def extract_belly_from_prediction(
    original: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int] = (80, 320),
    auto_rotate: bool = True,
    warp: bool = True,
) -> Image.Image:
    """
    original: np.ndarray HxWx3 (RGB)
    mask: np.ndarray HxW uint8 (0 or 255)
    """
    mask_gray = mask.astype(np.uint8)

    # Бинаризуем
    mask_bool = mask_gray > 0
    if not mask_bool.any():
        raise ValueError("Mask is empty — no belly detected")

    # Crop bounding box
    ys, xs = np.nonzero(mask_bool)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    # zero-out background
    masked = np.zeros_like(original)
    masked[mask_bool] = original[mask_bool]
    crop = masked[y0:y1, x0:x1]

    belly_img = Image.fromarray(crop)

    # Auto-rotate: make portrait
    if auto_rotate and belly_img.width > belly_img.height:
        belly_img = belly_img.rotate(90, expand=True)

    # Try warp
    if warp:
        warped = warp_belly_to_rect(np.array(belly_img), target_size)
        if warped is not None:
            return warped

    # fallback: plain resize
    return belly_img.resize(target_size, Image.BILINEAR)
