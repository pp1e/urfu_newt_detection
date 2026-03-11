from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from settings.embedder_loader import Embedder

@dataclass(frozen=True)
class ImageEmbedding:
    original: np.ndarray
    rotated: np.ndarray


def embed_pil_image(embedder: Embedder, img: Image.Image) -> np.ndarray:
    x = embedder.transform(img)  # CHW float32 tensor
    x = x.unsqueeze(0).to(embedder.device)

    with torch.inference_mode():
        emb = embedder.model.forward_embedding(x).detach().cpu().numpy().astype(np.float32)[0]
    return emb  # shape: (D,)



def embed_image_bytes(embedder: Embedder, image_bytes: bytes) -> ImageEmbedding:
    img_original = Image.open(BytesIO(image_bytes)).convert("RGB")
    emb_original = embed_pil_image(embedder, img_original)

    img_rot = img_original.rotate(180, expand=False)  # 180° вокруг центра
    emb_rotated = embed_pil_image(embedder, img_rot)

    return ImageEmbedding(
        original=emb_original,
        rotated=emb_rotated,
    )
