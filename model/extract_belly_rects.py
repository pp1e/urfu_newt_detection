from __future__ import annotations

"""
Apply predicted belly masks to source photos, crop the belly region, and
stretch it to a fixed rectangle (default 80x320).

Example:
    python scripts/extract_belly_rects.py \\
        --mask-dir output/test_belly_masks_karelin_run3 \\
        --images-dir data/source/karelin_newt_data-fixed \\
        --output-dir output/belly_rects_karelin_run3
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop bellies using masks and resize to a fixed rectangle.")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("output/test_belly_masks_karelin_run3"),
        help="Directory with predicted belly masks (_belly_mask.png).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/source/karelin_newt_data-fixed"),
        help="Directory with source photos (mirrors mask directory structure).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/belly_rects_karelin_run3"),
        help="Where to store stretched belly crops.",
    )
    parser.add_argument("--width", type=int, default=80, help="Target width of the stretched belly.")
    parser.add_argument("--height", type=int, default=320, help="Target height of the stretched belly.")
    parser.add_argument(
        "--no-auto-rotate",
        action="store_true",
        help="Disable auto-rotation that makes the crop portrait (taller than wide).",
    )
    parser.add_argument(
        "--no-warp",
        action="store_true",
        help="Disable per-row warp (falls back to plain resize of the cropped mask area).",
    )
    parser.add_argument(
        "--mask-suffix",
        type=str,
        default="_belly_mask.png",
        help="Filename suffix that denotes mask files.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_belly_rect",
        help="Suffix appended to output filenames before the extension.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for number of masks to process.")
    return parser.parse_args()


def iter_masks(mask_root: Path, mask_suffix: str) -> Iterable[Path]:
    pattern = f"*{mask_suffix}"
    yield from sorted(mask_root.rglob(pattern))


def find_source_image(mask_path: Path, mask_root: Path, images_root: Path) -> Path:
    rel = mask_path.relative_to(mask_root)
    base_stem = mask_path.stem
    if base_stem.endswith("_belly_mask"):
        base_stem = base_stem[: -len("_belly_mask")]

    candidates = []
    for ext in (".JPG", ".JPEG", ".PNG", ".jpg", ".jpeg", ".png"):
        candidate = images_root / rel.parent / f"{base_stem}{ext}"
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"Source image not found for mask: {mask_path}")
    return candidates[0]


def align_mask(mask_img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    if mask_img.size == target_size:
        return mask_img
    for angle in (90, -90):
        rotated = mask_img.rotate(angle, expand=True)
        if rotated.size == target_size:
            return rotated
    raise ValueError(f"Mask size {mask_img.size} does not match image size {target_size}")


def extract_and_resize(
    image_path: Path,
    mask_path: Path,
    target_size: tuple[int, int],
    auto_rotate: bool,
    warp: bool,
) -> Image.Image:
    with Image.open(image_path) as img:
        image_rgb = img.convert("RGB")
    with Image.open(mask_path) as mask_img:
        mask_gray = mask_img.convert("L")

    mask_gray = align_mask(mask_gray, image_rgb.size)

    image_np = np.array(image_rgb)
    mask_np = np.array(mask_gray)
    mask_bool = mask_np > 0
    if not mask_bool.any():
        raise ValueError(f"Mask is empty: {mask_path}")

    ys, xs = np.nonzero(mask_bool)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    masked = np.zeros_like(image_np)
    masked[mask_bool] = image_np[mask_bool]
    crop = masked[y0:y1, x0:x1]

    belly_img = Image.fromarray(crop)
    if auto_rotate and belly_img.width > belly_img.height:
        belly_img = belly_img.rotate(90, expand=True)

    if warp:
        warped = warp_belly_to_rect(np.array(belly_img), target_size)
        if warped is not None:
            return warped

    belly_img = belly_img.resize(target_size, Image.BILINEAR)
    return belly_img


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


def main() -> None:
    args = parse_args()
    mask_root = args.mask_dir.expanduser().resolve()
    images_root = args.images_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mask_paths = list(iter_masks(mask_root, args.mask_suffix))
    if args.limit is not None:
        mask_paths = mask_paths[: args.limit]

    if not mask_paths:
        raise RuntimeError(f"No mask files found under {mask_root}")

    target_size = (args.width, args.height)
    processed = 0
    skipped = 0

    for mask_path in mask_paths:
        try:
            src_image = find_source_image(mask_path, mask_root, images_root)
            belly_img = extract_and_resize(
                src_image,
                mask_path,
                target_size,
                auto_rotate=not args.no_auto_rotate,
                warp=not args.no_warp,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Skipping {mask_path} due to error: {exc}")
            skipped += 1
            continue

        rel = mask_path.relative_to(mask_root)
        base_stem = mask_path.stem
        if base_stem.endswith("_belly_mask"):
            base_stem = base_stem[: -len("_belly_mask")]
        out_name = f"{base_stem}{args.output_suffix}_{args.width}x{args.height}.jpg"
        out_path = output_root / rel.parent / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        belly_img.save(out_path, quality=95)
        processed += 1

    print(
        f"Done. Processed {processed} mask(s), "
        f"skipped {skipped}. Results saved to {output_root}"
    )


if __name__ == "__main__":
    main()
