import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import random

# === ПАРАМЕТРЫ ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = Path("dataset")
MASKS_DIR = Path("masks_manual")
MODEL_SAVE = Path("models/unet_belly_fine.pt")
BASE_MODEL = Path("models/unet_belly_scratch.pt")  # 🔹 fine-tune от предыдущей
BATCH_SIZE = 4
EPOCHS = 30
VAL_SPLIT = 0.2
IMG_SIZE = 512

# === КАСТОМНЫЕ ЛОССЫ ===
class FocalLoss(nn.Module):
    """Focal Loss для балансировки классов"""
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma
        return (self.alpha * focal_term * bce_loss).mean()


# === DATASET ===
class BellyDataset(Dataset):
    def __init__(self, images_root, masks_root, transform=None):
        self.transform = transform
        self.samples = []

        for mask_path in masks_root.rglob("*_mask.png"):
            rel_parts = mask_path.relative_to(masks_root).parts
            species = rel_parts[0]
            subdirs = rel_parts[1:-1]
            fname = rel_parts[-1]

            img_path = images_root / species
            for s in subdirs:
                img_path /= s
            img_path /= f"{Path(fname).stem.replace('_mask', '')}.JPG"

            if not img_path.exists():
                for ext in [".jpg", ".jpeg", ".JPEG", ".png"]:
                    alt = img_path.with_suffix(ext)
                    if alt.exists():
                        img_path = alt
                        break
                else:
                    continue

            mask = cv2.imread(str(mask_path), 0)
            if mask is None or mask.sum() == 0:
                continue

            self.samples.append((img_path, mask))

        print(f"📂 Найдено {len(self.samples)} валидных образцов")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask = self.samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        mask = (mask > 0.5).float().unsqueeze(0)
        return img, mask


# === АУГМЕНТАЦИИ (усиленные) ===
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.7),
    A.ColorJitter(p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
    A.RandomShadow(p=0.2),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=30, p=0.6),
    A.GaussianBlur(p=0.3),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2(),
])


# === ДАННЫЕ ===
dataset = BellyDataset(DATASET_DIR, MASKS_DIR, transform=train_tf)
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

print(f"📊 Train: {train_size}, Val: {val_size}")

# === МОДЕЛЬ ===
model = smp.Unet(
    encoder_name="resnet34",
    in_channels=3,
    classes=1,
    encoder_weights="imagenet",
)

if BASE_MODEL.exists():
    model.load_state_dict(torch.load(BASE_MODEL, map_location=DEVICE))
    print(f"✅ Загружена базовая модель: {BASE_MODEL.name}")

model.to(DEVICE)

# === ЛОССЫ ===
dice = smp.losses.DiceLoss(mode="binary")
focal = FocalLoss(alpha=0.8, gamma=2)
def mixed_loss(pred, target):
    return 0.6 * dice(pred, target) + 0.4 * focal(pred, target)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
best_iou = 0.0

# === ОБУЧЕНИЕ ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss = mixed_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)

    # === ВАЛИДАЦИЯ ===
    model.eval()
    ious = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = torch.sigmoid(model(imgs))
            preds = (preds > 0.5).float()
            intersection = (preds * masks).sum((1, 2, 3))
            union = (preds + masks - preds * masks).sum((1, 2, 3))
            iou = (intersection / (union + 1e-6)).mean().item()
            ious.append(iou)

    mean_iou = np.mean(ious)
    print(f"📈 Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | IoU: {mean_iou:.4f}")

    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), MODEL_SAVE)
        print(f"💾 Сохранена модель с IoU={best_iou:.4f}")

print("🏁 Обучение завершено.")
