from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent.parent
    )
)

from tg_bot.bot.belly_vectorization.classifier_embedder import ResNetClassifierEmbedder
from tg_bot.bot.belly_vectorization.build_transform import build_transforms


# -----------------------------
# Data
# -----------------------------

@dataclass(frozen=True)
class Sample:
    path: Path
    class_id: int
    class_name: str


class BellyIdDataset(Dataset):
    """
    Expects:
      root/
        1/
          *.jpg
          subfolders/.../*.jpg
        2/
        ...
        21 (error)/   <-- will be skipped by default
    """

    def __init__(
        self,
        root: Path,
        transform: nn.Module,
        *,
        min_images_per_class: int = 1,
        class_to_id: Dict[str, int] | None = None,
        allow_non_numeric: bool = True,
    ):
        self.root = root
        self.transform = transform

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        numeric_re = re.compile(r"^\d+$")

        # 1) Выбираем папки-классы
        class_dirs: List[Path] = []
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            name = p.name
            if allow_non_numeric:
                class_dirs.append(p)
            else:
                if numeric_re.match(name):
                    class_dirs.append(p)

        if not class_dirs:
            raise RuntimeError(f"No class folders found in {root}")

        # 2) Собираем изображения по классам (рекурсивно)
        class_to_paths: Dict[str, List[Path]] = {}
        for class_dir in class_dirs:
            class_name = class_dir.name
            paths = [
                img_path
                for img_path in class_dir.rglob("*")
                if img_path.is_file() and img_path.suffix.lower() in exts
            ]
            if len(paths) >= min_images_per_class:
                class_to_paths[class_name] = sorted(paths)

        if not class_to_paths:
            raise RuntimeError(
                f"No images found under {root} (after filtering). "
                f"Try lowering min_images_per_class or check extensions."
            )

        # 3) Маппинг class_name -> class_id (0..C-1)
        class_names = sorted(class_to_paths.keys(), key=lambda s: int(s) if s.isdigit() else s)
        if class_to_id is None:
            class_to_id = {name: i for i, name in enumerate(class_names)}
        self.class_to_id = class_to_id

        # 4) Финальный список сэмплов
        samples: List[Sample] = []
        for class_name in class_names:
            cid = class_to_id[class_name]
            for img_path in class_to_paths[class_name]:
                samples.append(Sample(path=img_path, class_id=cid, class_name=class_name))

        self.samples = samples
        self.class_names = class_names

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.transform(img)
        return x, s.class_id


# -----------------------------
# Training
# -----------------------------

@dataclass
class TrainConfig:
    data_root: Path
    output_path: Path
    image_size: int = 224
    embedding_dim: int = 256
    batch_size: int = 64
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 17
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_ratio: float = 0.15


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_long_on_device(y, device: torch.device) -> torch.Tensor:
    # Убираем warning "torch.tensor(tensor)"
    if isinstance(y, torch.Tensor):
        return y.to(device=device, dtype=torch.long)
    return torch.tensor(y, dtype=torch.long, device=device)


def train_embedder_classification(cfg: TrainConfig) -> None:
    _seed_all(cfg.seed)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf = build_transforms(cfg.image_size)

    # Создаём class_to_id по папкам
    class_names = sorted([p.name for p in cfg.data_root.iterdir() if p.is_dir()])
    if not class_names:
        raise RuntimeError(f"No class folders found in {cfg.data_root}")
    class_to_id = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_to_id)

    # Загружаем все образцы, потом делаем random split по индексам
    full_dataset_for_index = BellyIdDataset(cfg.data_root, transform=val_tf, class_to_id=class_to_id)
    n = len(full_dataset_for_index)
    indices = list(range(n))
    random.shuffle(indices)
    val_n = max(1, int(n * cfg.val_ratio))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    # Два датасета с разными transforms, но одинаковыми samples:
    train_dataset = BellyIdDataset(cfg.data_root, transform=train_tf, class_to_id=class_to_id)
    val_dataset = BellyIdDataset(cfg.data_root, transform=val_tf, class_to_id=class_to_id)

    class Subset(Dataset):
        def __init__(self, base: Dataset, keep: List[int]):
            self.base = base
            self.keep = keep
        def __len__(self): return len(self.keep)
        def __getitem__(self, i): return self.base[self.keep[i]]

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
    )

    device = torch.device(cfg.device)

    model = ResNetClassifierEmbedder(
        num_classes=num_classes,
        embedding_dim=cfg.embedding_dim,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = 0.0

    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        train_bar = tqdm(train_loader, desc=f"Train {epoch}/{cfg.epochs}", unit="batch", leave=False)
        for x, y in train_bar:
            x = x.to(device, non_blocking=True)
            y = _to_long_on_device(y, device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total += bs
            correct += int((logits.argmax(dim=1) == y).sum().item())

            train_bar.set_postfix(
                loss=f"{(total_loss / max(1, total)):.4f}",
                acc=f"{(correct / max(1, total)):.4f}",
            )

        scheduler.step()
        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # val
        model.eval()
        v_total = 0
        v_correct = 0
        val_bar = tqdm(val_loader, desc="Val", unit="batch", leave=False)
        with torch.inference_mode():
            for x, y in val_bar:
                x = x.to(device, non_blocking=True)
                y = _to_long_on_device(y, device)
                logits = model(x)
                v_total += x.size(0)
                v_correct += int((logits.argmax(dim=1) == y).sum().item())
                val_bar.set_postfix(acc=f"{(v_correct / max(1, v_total)):.4f}")

        val_acc = v_correct / max(1, v_total)
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.4f}",
            val_acc=f"{val_acc:.4f}",
        )

        # Сохраняем лучший чекпоинт (включая mapping классов)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_id": class_to_id,
                    "embedding_dim": cfg.embedding_dim,
                    "image_size": cfg.image_size,
                    "backbone": "resnet34",
                    "best_val_acc": best_val_acc,
                },
                cfg.output_path,
            )
            tqdm.write(f"Saved best checkpoint to {cfg.output_path} (val_acc={best_val_acc:.4f})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True, help="Папка датасета (root с папками классов)")
    p.add_argument("--run-name", type=str, required=True, help="Название прогона/инференса (для имени файла)")

    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                   help="cuda / cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_path = Path("output") / f"{args.run_name}.pt"

    cfg = TrainConfig(
        data_root=args.data_root,
        output_path=out_path,
        device=args.device,
    )

    train_embedder_classification(cfg)
