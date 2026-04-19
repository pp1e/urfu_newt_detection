"""
train.py — обучение SegFormer-B2 для fully automatic сегментации
брюшной поверхности тритонов.

Структура датасета:
    dataset/
        karelin-newt/
            1/
                photo.jpg
                annotations_vgg.json
            2/
                ...
        ribbed-newt/
            1/
                ...
            ...

Маски генерируются на лету из VGG JSON.

Пример запуска:
    python train.py --dataset-dir dataset --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".JPG",
    ".JPEG",
    ".PNG",
}

DEFAULT_ENCODER = "nvidia/segformer-b2-finetuned-ade-512-512"


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    image_path: Path
    polygon_points: list[tuple[int, int]]
    species: str  # "karelin" | "ribbed"


# ---------------------------------------------------------------------------
# Чтение датасета из VGG JSON
# ---------------------------------------------------------------------------

def parse_vgg_json(json_path: Path) -> dict[str, list[tuple[int, int]]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    result: dict[str, list[tuple[int, int]]] = {}
    for entry in data.values():
        filename = entry.get("filename", "")
        regions = entry.get("regions", {})
        regions_iter = regions.values() if isinstance(regions, dict) else regions

        for region in regions_iter:
            shape = region.get("shape_attributes", {})
            if shape.get("name") != "polygon":
                continue

            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if len(xs) < 3:
                continue

            result[filename] = [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]
            break

    return result


def collect_samples(dataset_dir: Path) -> list[Sample]:
    dataset_dir = dataset_dir.resolve()
    samples: list[Sample] = []

    for json_path in sorted(dataset_dir.rglob("annotations_vgg.json")):
        folder = json_path.parent
        rel_parts = folder.relative_to(dataset_dir).parts
        species = "karelin" if rel_parts and "karelin" in rel_parts[0].lower() else "ribbed"

        annotations = parse_vgg_json(json_path)
        if not annotations:
            continue

        for filename, points in annotations.items():
            image_path: Path | None = None

            for ext in IMAGE_EXTENSIONS:
                candidate = folder / Path(filename).with_suffix(ext).name
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                candidate = folder / filename
                if candidate.exists():
                    image_path = candidate

            if image_path is None:
                continue

            samples.append(
                Sample(
                    image_path=image_path,
                    polygon_points=points,
                    species=species,
                )
            )

    if not samples:
        raise RuntimeError(
            f"Не найдено ни одного образца в {dataset_dir}. Проверьте структуру папок и наличие annotations_vgg.json."
        )

    return samples


def make_mask_from_polygon(
    points: list[tuple[int, int]],
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


# ---------------------------------------------------------------------------
# Разбиение на выборки
# ---------------------------------------------------------------------------

def stratified_split(
    samples: list[Sample],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 17,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(seed)

    by_species: dict[str, list[Sample]] = {}
    for sample in samples:
        by_species.setdefault(sample.species, []).append(sample)

    train_all: list[Sample] = []
    val_all: list[Sample] = []
    test_all: list[Sample] = []

    for species_samples in by_species.values():
        current = list(species_samples)
        rng.shuffle(current)

        n = len(current)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))

        test_all.extend(current[:n_test])
        val_all.extend(current[n_test : n_test + n_val])
        train_all.extend(current[n_test + n_val :])

    return train_all, val_all, test_all


def save_split(
    train: list[Sample],
    val: list[Sample],
    test: list[Sample],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "train": [str(sample.image_path) for sample in train],
        "val": [str(sample.image_path) for sample in val],
        "test": [str(sample.image_path) for sample in test],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Разбиение сохранено: {path}")


def load_split_or_create(
    all_samples: list[Sample],
    split_cache: Path,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    sample_by_path = {str(sample.image_path): sample for sample in all_samples}

    if split_cache.exists():
        print(f"Загружаем разбиение из кэша: {split_cache}")
        with open(split_cache, encoding="utf-8") as f:
            data = json.load(f)

        train = [sample_by_path[p] for p in data["train"] if p in sample_by_path]
        val = [sample_by_path[p] for p in data["val"] if p in sample_by_path]
        test = [sample_by_path[p] for p in data["test"] if p in sample_by_path]
    else:
        train, val, test = stratified_split(all_samples, seed=seed)
        save_split(train, val, test, split_cache)

    return train, val, test


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NewtDataset(Dataset):
    def __init__(self, samples: list[Sample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        with Image.open(sample.image_path) as pil_img:
            image = np.array(ImageOps.exif_transpose(pil_img).convert("RGB"))

        h, w = image.shape[:2]
        mask = make_mask_from_polygon(sample.polygon_points, h, w).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image_t = aug["image"]
            mask_t = aug["mask"].unsqueeze(0)
        else:
            image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_t = torch.from_numpy(mask).unsqueeze(0)

        return image_t, mask_t


# ---------------------------------------------------------------------------
# Аугментации
# ---------------------------------------------------------------------------

def build_transforms(img_size: int):
    normalize = A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
    )

    train_tf = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.05, 0.05),
                rotate=(-25, 25),
                border_mode=cv2.BORDER_REFLECT101,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.4),
            A.ColorJitter(p=0.3),
            A.GaussianBlur(blur_limit=5, p=0.2),
            normalize,
            ToTensorV2(),
        ]
    )

    val_tf = A.Compose(
        [
            A.Resize(img_size, img_size),
            normalize,
            ToTensorV2(),
        ]
    )

    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

def build_model(
    encoder: str,
    device: torch.device,
) -> nn.Module:
    model = SegformerForSemanticSegmentation.from_pretrained(
        encoder,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    return model.to(device)


def forward_logits(
    model: nn.Module,
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


# ---------------------------------------------------------------------------
# Метрики и loss
# ---------------------------------------------------------------------------

def dice_score(
    preds_binary: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    p = preds_binary.reshape(preds_binary.size(0), -1)
    t = targets.reshape(targets.size(0), -1)
    inter = (p * t).sum(dim=1)
    return float(((2 * inter + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)).mean())


def iou_score(
    preds_binary: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    p = preds_binary.reshape(preds_binary.size(0), -1)
    t = targets.reshape(targets.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - inter
    return float(((inter + eps) / (union + eps)).mean())


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# Обучение и оценка
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    img_size: int,
) -> float:
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = forward_logits(model, images, img_size)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(1, len(loader))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    threshold: float,
    img_size: int,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    dice_vals: list[float] = []
    iou_vals: list[float] = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = forward_logits(model, images, img_size)
            total_loss += criterion(logits, masks).item()

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            dice_vals.append(dice_score(preds, masks))
            iou_vals.append(iou_score(preds, masks))

    mean_loss = total_loss / max(1, len(loader))
    return mean_loss, float(np.mean(dice_vals)), float(np.mean(iou_vals))


# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

class CSVLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou"])

    def log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_dice: float,
        val_iou: float,
    ) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_dice:.6f}",
                    f"{val_iou:.6f}",
                ]
            )


# ---------------------------------------------------------------------------
# Аргументы
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение SegFormer-B2 для сегментации брюшной поверхности тритонов")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Корневая папка датасета",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=DEFAULT_ENCODER,
        help="Hugging Face имя базового SegFormer",
    )
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.6,
        help="Вес Dice loss в комбинированной функции потерь",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Папка для чекпоинтов и логов",
    )
    parser.add_argument(
        "--split-cache",
        type=Path,
        default=Path("output/split.json"),
        help="JSON с сохранённым train/val/test split",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = args.encoder

    print(f"Устройство: {device}")
    print(f"Модель: segformer | Энкодер: {encoder}")

    print("Читаем датасет...")
    all_samples = collect_samples(args.dataset_dir)
    print(f"Всего образцов: {len(all_samples)}")

    train_samples, val_samples, test_samples = load_split_or_create(
        all_samples,
        args.split_cache,
        args.seed,
    )
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    train_tf, val_tf = build_transforms(args.img_size)

    train_loader = DataLoader(
        NewtDataset(train_samples, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        NewtDataset(val_samples, val_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(encoder, device)

    bce_loss_fn = nn.BCEWithLogitsLoss()

    def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            args.dice_weight * soft_dice_loss(logits, targets)
            + (1 - args.dice_weight) * bce_loss_fn(logits, targets)
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    run_name = "segformer_nvidia_mit-b2"
    checkpoint_path = args.output_dir / f"{run_name}_best.pt"
    log_path = args.output_dir / "logs" / f"{run_name}.csv"
    logger = CSVLogger(log_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Чекпоинт: {checkpoint_path}")
    print(f"Лог: {log_path}")

    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.img_size,
        )
        scheduler.step()

        val_loss, val_dice, val_iou = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.threshold,
            args.img_size,
        )

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_iou={val_iou:.4f}"
        )

        logger.log(epoch, train_loss, val_loss, val_dice, val_iou)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "model_name": "segformer",
                    "encoder": encoder,
                    "img_size": args.img_size,
                    "threshold": args.threshold,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  → Лучшая модель сохранена (Dice={best_dice:.4f})")

    print(f"\nОбучение завершено. Лучший Dice на val: {best_dice:.4f}")


if __name__ == "__main__":
    main()
