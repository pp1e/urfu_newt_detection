"""Train a U-Net to segment newt bellies using manual masks.

By default the script expects:
- original photos under ``data/karelin_newt_labeled`` (same folder naming as masks)
- binary masks exported to ``masks_manual``. Every mask file should end with ``_mask``.

Usage example:
    python train_manual_belly_unet.py \
        --images-dir data/karelin_newt_labeled \
        --masks-dir masks_manual/karelin_newt_labeled \
        --epochs 40 --batch-size 6
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net belly segmenter")
    parser.add_argument("--images-dir", type=Path, default=Path("data/karelin_newt_labeled"),
                        help="Root directory with original images")
    parser.add_argument("--masks-dir", type=Path, default=Path("masks_manual"),
                        help="Directory with manually labeled masks")
    parser.add_argument("--output", type=Path, default=Path("output/segmentation_all_kinds.pt"),
                        help="Path to save the best checkpoint")
    parser.add_argument("--img-size", type=int, default=768, help="Image resize to feed the network")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of samples used for validation")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="Backbone from segmentation_models_pytorch")
    parser.add_argument("--encoder-weights", type=str, default="imagenet",
                        help="Weights for the encoder (set to None for random init)")
    parser.add_argument("--dice-weight", type=float, default=0.6,
                        help="Weight of Dice loss vs BCE")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binarizing predictions when measuring IoU")
    parser.add_argument("--min-mask-coverage", type=float, default=0.001,
                        help="Minimal fraction of positive pixels to keep the mask")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit of samples for debugging")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _iter_candidate_dirs(rel_parts: Sequence[str]) -> Iterable[Path]:
    # try keeping every suffix of the path (so we can drop dataset name if needed)
    if not rel_parts:
        yield Path()
        return
    for start in range(len(rel_parts)):
        slice_parts = rel_parts[start:-1]
        yield Path(*slice_parts) if slice_parts else Path()


def collect_samples(images_root: Path, masks_root: Path, min_coverage: float,
                    limit: int | None = None) -> List[Sample]:
    images_root = images_root.expanduser().resolve()
    masks_root = masks_root.expanduser().resolve()
    if not images_root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    if not masks_root.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_root}")

    exts = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".tif", ".tiff")
    samples: List[Sample] = []

    mask_files = sorted(p for p in masks_root.rglob("*_mask.*") if "overlay" not in p.name)
    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask_binary = (mask > 0).astype(np.uint8)
        coverage = mask_binary.mean()
        if coverage < min_coverage:
            continue

        rel_parts = mask_path.relative_to(masks_root).parts
        stem = mask_path.stem[:-5] if mask_path.stem.endswith("_mask") else mask_path.stem

        found_image: Path | None = None
        for candidate_dir in _iter_candidate_dirs(rel_parts):
            for ext in exts:
                candidate = images_root / candidate_dir / f"{stem}{ext}"
                if candidate.exists():
                    found_image = candidate
                    break
            if found_image is not None:
                break

        if found_image is None:
            continue

        samples.append(Sample(found_image, mask_path))
        if limit is not None and len(samples) >= limit:
            break

    if not samples:
        raise RuntimeError("No matching image-mask pairs were found. Check directory arguments.")

    return samples


class BellySegmentationDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = np.array(Image.open(sample.image_path).convert("RGB"))
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {sample.mask_path}")
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def build_transforms(img_size: int):
    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=25, border_mode=cv2.BORDER_REFLECT101, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.ColorJitter(p=0.3),
        A.GaussianBlur(blur_limit=5, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    return train_tf, val_tf


def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    preds = preds.contiguous().view(preds.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    return ((2 * intersection + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)).mean()


def iou_score(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    preds = preds.contiguous().view(preds.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device,
             threshold: float) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            dice_scores.append(dice_coefficient(preds, masks).item())
            iou_scores.append(iou_score(preds, masks).item())

    mean_loss = total_loss / max(1, len(loader))
    return mean_loss, float(np.mean(dice_scores)), float(np.mean(iou_scores))


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = collect_samples(args.images_dir, args.masks_dir,
                              min_coverage=args.min_mask_coverage,
                              limit=args.limit)
    if len(samples) < 2:
        raise RuntimeError("Need at least two samples to run train/validation split.")
    random.shuffle(samples)
    val_count = max(1, int(len(samples) * args.val_split))
    train_samples = samples[val_count:]
    val_samples = samples[:val_count]
    print(f"Found {len(samples)} samples (train={len(train_samples)}, val={len(val_samples)})")

    train_tf, val_tf = build_transforms(args.img_size)
    train_dataset = BellySegmentationDataset(train_samples, train_tf)
    val_dataset = BellySegmentationDataset(val_samples, val_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None if args.encoder_weights == "None" else args.encoder_weights,
        in_channels=3,
        classes=1
    ).to(device)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()

    def criterion(logits, targets):
        return args.dice_weight * dice_loss(logits, targets) + (1 - args.dice_weight) * bce(logits, targets)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device, args.threshold)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "args": vars(args),
            }, args.output)
            print(f"Saved new best model to {args.output} (Dice={best_dice:.4f})")

    print("Training finished")


if __name__ == "__main__":
    main()
