import torch
import cv2
import numpy as np
from settings.loader import MODEL_KARELINA, MODEL_RIBBED, DEVICE
import albumentations as A
from albumentations.pytorch import ToTensorV2

# === Препроцесс ===
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])


# ======================================================
# 1) Overlay (как раньше)
# ======================================================

def apply_mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    color = (0, 255, 0)  # зеленый
    alpha = 0.4

    mask_color = np.zeros_like(image)
    mask_color[mask == 1] = color

    blended = cv2.addWeighted(overlay, 1, mask_color, alpha, 0)
    return blended


def extract_belly_patch(image, mask, target_height=1000):
    """
    Чистый crop брюшка:
    - убраны белые квадраты (альфа корректна)
    - выровнено и вытянуто по вертикали
    - голова сверху
    """

    cnts, _ = cv2.findContours((mask * 255).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    angle = rect[2]
    if angle < -45:
        angle += 90

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    rot_img = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    rot_mask = cv2.warpAffine(
        mask.astype(np.uint8), M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # tight crop
    ys, xs = np.where(rot_mask == 1)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = rot_img[y1:y2 + 1, x1:x2 + 1]
    crop_mask = rot_mask[y1:y2 + 1, x1:x2 + 1]

    # Проверка ориентации (где голова)
    top_width = np.sum(crop_mask[: int(crop_mask.shape[0] * 0.1), :])
    bottom_width = np.sum(crop_mask[int(crop_mask.shape[0] * 0.9):, :])
    if bottom_width < top_width:
        crop = cv2.rotate(crop, cv2.ROTATE_180)
        crop_mask = cv2.rotate(crop_mask, cv2.ROTATE_180)

    # Контраст CLAHE
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # RGBA + альфа
    crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    crop_rgba[crop_mask == 0, 3] = 0

    # Морфология на альфе — удаляем рамку
    alpha = crop_rgba[:, :, 3]
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.erode(alpha, kernel, iterations=1)
    alpha = cv2.medianBlur(alpha, 3)
    crop_rgba[:, :, 3] = alpha

    # Вытягиваем по вертикали
    cur_h, cur_w = crop_rgba.shape[:2]
    aspect = cur_h / cur_w
    if aspect < 2.2:
        fy = 2.2 / aspect
        crop_rgba = cv2.resize(crop_rgba, None, fx=1, fy=fy, interpolation=cv2.INTER_CUBIC)

    # Ресайз по высоте
    cur_h, cur_w = crop_rgba.shape[:2]
    scale = target_height / cur_h
    new_w = int(cur_w * scale)
    resized = cv2.resize(crop_rgba, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    # tight crop по альфа
    alpha = resized[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        resized = resized[y1:y2 + 1, x1:x2 + 1]

    return resized





# ======================================================
# 3) Основная функция predict_mask (overlay + belly)
# ======================================================

def predict_mask(image_bytes: bytes, model_type: str):
    """
    Возвращает:
    overlay_jpg_bytes,
    belly_png_bytes
    """

    # === выбор модели ===
    if model_type == "karelina":
        model = MODEL_KARELINA
    else:
        model = MODEL_RIBBED

    # === загрузка изображения ===
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig_h, orig_w = img.shape[:2]

    # === препроцесс ===
    transformed = transform(image=img)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    # === инференс ===
    with torch.no_grad():
        pred = model(tensor)

    # === маска ===
    if pred.shape[1] == 1:
        pred = torch.sigmoid(pred)
        mask = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
    else:
        pred = torch.softmax(pred, dim=1)
        mask = (pred[0, 1].cpu().numpy() > 0.5).astype(np.uint8)

    # === ресайз маски под оригинал ===
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # === overlay ===
    overlay = apply_mask_overlay(img, mask_resized)
    _, overlay_buf = cv2.imencode(".jpg", overlay)

    # === вырезаем брюшко ===
    belly = extract_belly_patch(img, mask_resized)
    if belly is None:
        belly = np.zeros((300, 150, 4), dtype=np.uint8)

    _, belly_buf = cv2.imencode(".png", belly)

    return overlay_buf.tobytes(), belly_buf.tobytes()
