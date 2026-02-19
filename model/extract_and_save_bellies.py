from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent
    )
)

from tg_bot.bot.belly_detection.extract_belly import extract_belly_from_prediction
from tg_bot.bot.belly_detection.clean_mask import clean_mask

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
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop bellies using masks and resize to a fixed rectangle.")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help="Directory with predicted belly masks (_belly_mask.png).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Directory with source photos (mirrors mask directory structure).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
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

    for angle in (90, -90, 180):
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

    mask_np = clean_mask(
        mask_np,
        min_area=600,
        kernel_size=9,
    )

    return extract_belly_from_prediction(
        original=image_np,
        mask=mask_np,
        target_size=target_size,
        auto_rotate=auto_rotate,
        warp=warp,
    )


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
