import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt

# === НАСТРОЙКИ ===
MODEL_PATH = Path("models/unet_belly_fine.pt")
DATASETS = ["dataset/triton_karelina", "dataset/rebristii_triton"]
OUT_ROOT = Path("predictions/belly_scratch_all")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
THRESHOLD = 0.35  # мягкий порог 0.3–0.4 даёт более плавные контуры

# === МОДЕЛЬ ===
print(f"🧠 Загружаем модель: {MODEL_PATH.name}")
model = smp.Unet(
    encoder_name="resnet34",
    in_channels=3,
    classes=1,
    encoder_weights="imagenet",
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === ТРАНСФОРМ ===
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2(),
])

# === ФУНКЦИЯ ПРЕДИКТА ===
def predict_image(image_path: Path, out_dir: Path):
    img = np.array(Image.open(image_path).convert("RGB"))
    orig_h, orig_w = img.shape[:2]

    transformed = transform(image=img)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE, dtype=torch.float32)

    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    mask = (pred > THRESHOLD).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (orig_w, orig_h))

    # === СОЗДАЁМ ОВЕРЛЕЙ ===
    overlay = img.copy()
    color_mask = np.zeros_like(img)
    color_mask[..., 1] = 255  # зелёный канал
    overlay = cv2.addWeighted(color_mask, 0.4, overlay, 0.6, 0)
    overlay[mask_resized == 0] = img[mask_resized == 0]

    mask_path = out_dir / f"{image_path.stem}_mask.png"
    overlay_path = out_dir / f"{image_path.stem}_overlay.jpg"

    cv2.imwrite(str(mask_path), mask_resized)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return img, overlay, mask_resized


# === ПРОХОД ПО ВСЕМ ПАПКАМ ===
for ds_root in DATASETS:
    ds_root = Path(ds_root)
    ds_name = ds_root.name
    print(f"\n📂 Обрабатываем датасет: {ds_name}")

    for subfolder in sorted(ds_root.iterdir()):
        if not subfolder.is_dir():
            continue

        imgs = sorted(list(subfolder.glob("*.JPG"))) + sorted(list(subfolder.glob("*.jpg")))
        if not imgs:
            continue

        out_dir = OUT_ROOT / ds_name / subfolder.name
        out_dir.mkdir(parents=True, exist_ok=True)

        all_overlays = []

        print(f"  🔸 {subfolder.name} ({len(imgs)} изображений)")
        for img_path in tqdm(imgs, desc=f"{ds_name}/{subfolder.name}"):
            _, overlay, _ = predict_image(img_path, out_dir)
            # уменьшить для коллажа
            overlay_small = cv2.resize(overlay, (256, 256))
            all_overlays.append(overlay_small)

        # === СОЗДАЁМ КОЛЛАЖ ===
        if all_overlays:
            cols = min(5, len(all_overlays))
            rows = int(np.ceil(len(all_overlays) / cols))
            canvas = np.ones((rows * 256, cols * 256, 3), dtype=np.uint8) * 255

            for idx, ov in enumerate(all_overlays):
                r, c = divmod(idx, cols)
                y1, y2 = r * 256, (r + 1) * 256
                x1, x2 = c * 256, (c + 1) * 256
                canvas[y1:y2, x1:x2] = ov

            collage_path = out_dir / f"preview_{subfolder.name}.png"
            cv2.imwrite(str(collage_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            print(f"  📸 Коллаж сохранён: {collage_path.relative_to(OUT_ROOT)}")

print(f"\n✅ Все результаты сохранены в {OUT_ROOT}")
