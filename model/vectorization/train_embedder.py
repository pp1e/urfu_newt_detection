from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent.parent
    )
)

from model.vectorization.datasets import build_datasets
from tg_bot.bot.belly_vectorization.constants import IMAGE_SIZE
from tg_bot.bot.belly_vectorization.classifier_embedder import DinoViTClassifierEmbedder
from tg_bot.bot.belly_vectorization.build_transform import build_transforms


# -----------------------------
# Training
# -----------------------------

@dataclass
class TrainConfig:
    data_root: Path
    output_path: Path
    embedding_dim: int = 256
    batch_size: int = 32
    epochs: int = 40
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 17
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_ratio: float = 0.15

    head_lr: float = 3e-4
    backbone_lr: float = 1e-5


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

    train_tf, val_tf = build_transforms(IMAGE_SIZE)

    # Загружаем все образцы, потом делаем random split по индексам
    datasets = build_datasets(
        data_root=cfg.data_root,
        train_tf=train_tf,
        val_tf=val_tf,
        val_ratio=cfg.val_ratio,
    )


    train_loader = DataLoader(
        datasets.train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
    )
    val_loader = DataLoader(
        datasets.val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
    )

    device = torch.device(cfg.device)

    model = DinoViTClassifierEmbedder(
        num_classes=len(datasets.class_to_id),
        embedding_dim=cfg.embedding_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg.backbone_lr},
            {"params": model.embedding_head.parameters(), "lr": cfg.head_lr},
            {"params": model.classifier.parameters(), "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
    )

    best_val_acc = 0.0

    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        train_bar = tqdm(train_loader, desc=f"Train {epoch}/{cfg.epochs}", unit="batch", leave=False)
        for x, y, _ in train_bar:
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
            for x, y, _ in val_bar:
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
                    "class_to_id": datasets.class_to_id,
                    "embedding_dim": cfg.embedding_dim,
                    "image_size": IMAGE_SIZE,
                    "backbone": "vit_small_patch14_dinov2",
                    "model_name": "vit_small_patch14_dinov2",
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
