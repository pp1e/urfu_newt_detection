import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A

MODEL_PATH = "output/model.pth"
IMG_PATH = "dataset/images/1/01-01-1224.jpg"
OUT_PATH = "output/01-01-1224.png"

IMG_SIZE = 512

# --- Подготовка ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0, 0, 0), std=(255, 255, 255)),
    ToTensorV2()
])

# --- Загрузка изображения ---
image = cv2.imread(IMG_PATH)
orig_h, orig_w = image.shape[:2]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

aug = transform(image=image_rgb)
tensor = aug["image"].unsqueeze(0).to(device)

# --- Предсказание ---
with torch.no_grad():
    pred = torch.sigmoid(model(tensor))[0][0].cpu().numpy()

mask = (pred > 0.5).astype(np.uint8) * 255
mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

# --- Сохранение ---
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
cv2.imwrite(OUT_PATH, mask_resized)
print(f"✅ Маска сохранена в {OUT_PATH}")
