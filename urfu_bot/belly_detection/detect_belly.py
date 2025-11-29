from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from urfu_bot.belly_detection.create_overlay import create_overlay
from urfu_bot.belly_detection.extract_belly import extract_belly_from_prediction
from urfu_bot.belly_detection.predict_mask import predict_mask

@dataclass
class BellyDetectionResult:
    belly: bytes
    overlay: bytes


def pil_to_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    buf = BytesIO()
    image.save(buf, format=image_format)
    return buf.getvalue()


def detect_belly(
    image_bytes: bytes,
    model_type: str,
    target_size=(80, 320),
    auto_rotate=True,
    warp=True,
    image_format: str = "PNG",
) -> BellyDetectionResult:
    mask_with_original = predict_mask(
        image_bytes=image_bytes,
        model_type=model_type,
    )

    belly_image = extract_belly_from_prediction(
        original=mask_with_original.original,
        mask=mask_with_original.mask,
        target_size=target_size,
        auto_rotate=auto_rotate,
        warp=warp,
    )

    overlay_image = create_overlay(
        original=mask_with_original.original,
        mask=mask_with_original.mask,
        alpha=0.4,
    )

    return BellyDetectionResult(
        belly=pil_to_bytes(
            image=belly_image,
            image_format=image_format,
        ),
        overlay=pil_to_bytes(
            overlay_image,
            image_format=image_format,
        ),
    )
