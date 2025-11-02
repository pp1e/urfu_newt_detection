import os
from pathlib import Path

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# --- Параметры ---
DATA_DIR = "dataset"
IMG_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")
MODEL_PATH = "output/model.pth"

IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-4

# --- Датасет ---
class TritonDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        img_dir_as_path = Path(img_dir)
        self.filepaths = [
            str(file) for file in img_dir_as_path.rglob("*")
                if file.is_file() and str(file).lower().endswith((".jpg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        # img_path = os.path.join(self.img_dir, img_name)
        mask_path = img_path.replace(".jpg", ".png").replace("images", "masks")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)  # бинаризация

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)

        return image, mask

# --- Аугментации ---
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    ToTensorV2()
])

# --- Загрузка данных ---
dataset = TritonDataset(IMG_DIR, MASK_DIR, transform=train_transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# --- Модель U-Net ---
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)

platform = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using platform: {platform}")
device = torch.device(platform)
model = model.to(device)

# --- Оптимизатор и лосс ---
loss_fn = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Обучение ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(device).float() / 255.0
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

# --- Сохранение модели ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Модель сохранена: {MODEL_PATH}")
