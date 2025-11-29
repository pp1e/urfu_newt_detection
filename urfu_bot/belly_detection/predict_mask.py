from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from numpy import ndarray

from settings.loader import MODEL_KARELINA, MODEL_RIBBED, IMG_SIZE, THRESHOLD, DEVICE
import albumentations as A
from albumentations.pytorch import ToTensorV2

@dataclass
class MaskWithOriginal:
    mask: ndarray
    original: ndarray


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


def prepare_image(image_bytes: bytes, transform: A.Compose) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(BytesIO(image_bytes)) as img:
        image = np.array(img.convert("RGB"))

    augmented = transform(image=image)
    tensor = augmented["image"]
    original = image

    tensor = tensor.unsqueeze(0)

    return tensor, original


def postprocess_prediction(
        pred: np.ndarray,
        original_shape: tuple[int, int],
        threshold: float,
) -> np.ndarray:
    h, w = original_shape
    pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (pred_resized >= threshold).astype(np.uint8) * 255
    return binary


def predict_mask(image_bytes: bytes, model_type: str) -> MaskWithOriginal:
    if model_type == "karelina":
        model = MODEL_KARELINA
    else:
        model = MODEL_RIBBED


    transform = build_transform(IMG_SIZE)

    batch_tensor, original = prepare_image(image_bytes, transform)
    batch_tensor = batch_tensor.to(DEVICE)
    with torch.no_grad():
        logits = model(batch_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    prob = probs[0]

    return MaskWithOriginal(
        mask = postprocess_prediction(
            prob[0],
            original.shape[:2],
            THRESHOLD,
        ),
        original = original,
    )
