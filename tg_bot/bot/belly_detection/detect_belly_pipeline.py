from dataclasses import dataclass, replace
from io import BytesIO

from PIL import Image

from bot.belly_detection.clean_mask import clean_mask
from bot.belly_detection.create_overlay import create_overlay
from bot.belly_detection.extract_belly import extract_belly_from_prediction
from bot.belly_detection.predict_mask import predict_mask

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
    target_size=(80, 320),
    auto_rotate=True,
    warp=True,
    image_format: str = "PNG",
) -> BellyDetectionResult:
    mask_with_original = predict_mask(
        image_bytes=image_bytes,
    )

    mask_with_original = replace(
        mask_with_original,
        mask=clean_mask(
            mask_with_original.mask,
            min_area=600,
            kernel_size=9,
        )
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
