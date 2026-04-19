"""Generate belly segmentation masks with SegFormer-B2.

The script loads a checkpoint produced by VKR training and
runs it over every image inside ``images-dir`` (recursively). For each source
photo both a binary mask (0/255) and an overlay (RGB photo with the mask
painted on top) are written to the output directory, preserving the
sub-directory layout. Optionally an extra overlay directory can be specified
to keep visualizations separately.

Example:
    python predict_manual_belly_segformer.py \
        --images-dir data/karelin_newt_labeled-fixed \
        --checkpoint weights/segformer_nvidia_mit-b2_best.pt \
        --output-dir output/belly_masks \
        --overlay-dir output/belly_overlays
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence
import math

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with SegFormer belly segmenter")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/karelin_newt_annotated-fixed"),
        help="Root directory with source images (will be scanned recursively)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/segformer_nvidia_mit-b2_best.pt"),
        help="Path to the trained SegFormer checkpoint (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/belly_predictions"),
        help="Where to store the predicted binary masks",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Optional extra directory for RGB overlays (keeps source layout)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images to process simultaneously",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Resize that feeds the network (falls back to checkpoint args, else 512)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold used to binarize the mask (defaults to checkpoint args or 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device spec (cuda, cpu, cuda:0, ...). 'auto' picks CUDA if available",
    )
    parser.add_argument(
        "--mask-suffix",
        type=str,
        default="_belly_mask",
        help="Suffix appended to the source stem when saving masks",
    )
    parser.add_argument(
        "--overlay-suffix",
        type=str,
        default="_overlay",
        help="Suffix used for overlay files when --overlay-dir is set",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Blending factor for overlay visualization",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of processed images (debug)",
    )
    return parser.parse_args()


def discover_images(images_root: Path, limit: int | None = None) -> List[Path]:
    images_root = images_root.expanduser().resolve()
    if not images_root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")

    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    result = [p for p in sorted(images_root.rglob("*")) if p.suffix.lower() in extensions]
    if limit is not None:
        result = result[:limit]
    if not result:
        raise RuntimeError(f"No images found under {images_root}")
    return result


def load_checkpoint(checkpoint_path: Path) -> tuple[dict, dict]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state'")
    saved_args = checkpoint.get("args", {})
    return checkpoint, saved_args


def resolve_arg(value, saved_args: dict, key: str, default):
    if value is not None:
        return value
    if key in saved_args and saved_args[key] is not None:
        return saved_args[key]
    return default


def build_transform(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def chunked(seq: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    size = max(1, size)
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


def prepare_batch(paths: Sequence[Path], transform: A.Compose) -> tuple[torch.Tensor, List[np.ndarray]]:
    tensors = []
    originals: List[np.ndarray] = []
    for path in paths:
        with Image.open(path) as img:
            image = np.array(ImageOps.exif_transpose(img).convert("RGB"))
        augmented = transform(image=image)
        tensors.append(augmented["image"])
        originals.append(image)
    batch = torch.stack(tensors, dim=0)
    return batch, originals


def postprocess_prediction(pred: np.ndarray, original_shape: tuple[int, int], threshold: float) -> np.ndarray:
    h, w = original_shape
    pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (pred_resized >= threshold).astype(np.uint8) * 255
    return binary


def make_overlay(image_rgb: np.ndarray, mask_binary: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = image_rgb.copy()
    mask_bool = mask_binary > 0
    highlight = np.zeros_like(image_rgb)
    highlight[..., 2] = 255  # red channel in RGB
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool].astype(np.float32) +
        alpha * highlight[mask_bool].astype(np.float32)
    ).astype(np.uint8)
    return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)


def save_mask(mask: np.ndarray, src_path: Path, src_root: Path, out_root: Path, suffix: str) -> Path:
    relative = src_path.relative_to(src_root)
    destination = out_root / relative.parent
    destination.mkdir(parents=True, exist_ok=True)
    mask_path = destination / f"{relative.stem}{suffix}.png"
    cv2.imwrite(str(mask_path), mask)
    return mask_path


def save_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    src_path: Path,
    src_root: Path,
    overlay_root: Path,
    suffix: str,
    alpha: float,
) -> Path:
    relative = src_path.relative_to(src_root)
    destination = overlay_root / relative.parent
    destination.mkdir(parents=True, exist_ok=True)
    overlay_image = make_overlay(image_rgb, mask, alpha)
    overlay_path = destination / f"{relative.stem}{suffix}.png"
    cv2.imwrite(str(overlay_path), overlay_image)
    return overlay_path


def build_model(device: torch.device) -> SegformerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    return model.to(device)


def forward_logits(
    model: SegformerForSemanticSegmentation,
    images: torch.Tensor,
    img_size: int,
) -> torch.Tensor:
    outputs = model(pixel_values=images)
    logits = outputs.logits
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)

    logits = F.interpolate(
        logits,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    return logits


def main() -> None:
    args = parse_args()

    checkpoint, saved_args = load_checkpoint(args.checkpoint)
    img_size = int(resolve_arg(args.img_size, saved_args, "img_size", 512))
    threshold = float(resolve_arg(args.threshold, saved_args, "threshold", 0.5))

    device_str = args.device.lower()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Loading SegFormer-B2: img_size={img_size} device={device}")

    model = build_model(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()

    images_root = args.images_dir.expanduser().resolve()
    masks_root = args.output_dir.expanduser().resolve()
    masks_root.mkdir(parents=True, exist_ok=True)
    overlay_root = args.overlay_dir.expanduser().resolve() if args.overlay_dir else None
    if overlay_root is not None:
        overlay_root.mkdir(parents=True, exist_ok=True)

    images = discover_images(images_root, args.limit)
    transform = build_transform(img_size)

    saved_masks = 0
    total_batches = math.ceil(len(images) / max(1, args.batch_size))
    for batch_paths in tqdm(
        chunked(images, args.batch_size),
        total=total_batches,
        desc="Predicting",
        unit="batch",
    ):
        batch_tensor, originals = prepare_batch(batch_paths, transform)
        batch_tensor = batch_tensor.to(device)

        with torch.no_grad():
            logits = forward_logits(model, batch_tensor, img_size)
            probs = torch.sigmoid(logits).cpu().numpy()

        for prob, original, path in zip(probs, originals, batch_paths):
            mask = postprocess_prediction(prob[0], original.shape[:2], threshold)
            save_mask(mask, path, images_root, masks_root, args.mask_suffix)
            save_overlay(
                original,
                mask,
                path,
                images_root,
                masks_root,
                args.overlay_suffix,
                args.overlay_alpha,
            )
            if overlay_root is not None and overlay_root != masks_root:
                save_overlay(
                    original,
                    mask,
                    path,
                    images_root,
                    overlay_root,
                    args.overlay_suffix,
                    args.overlay_alpha,
                )
            saved_masks += 1

    print(f"Saved {saved_masks} mask(s) and overlays to {masks_root}")
    if overlay_root is not None and overlay_root != masks_root:
        print(f"Additional overlays stored under {overlay_root}")


if __name__ == "__main__":
    main()
