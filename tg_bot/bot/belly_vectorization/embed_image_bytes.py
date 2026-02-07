from io import BytesIO

import numpy as np
import torch
from PIL import Image

from settings.embedder_loader import Embedder


def embed_image_bytes(embedder: Embedder, image_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = embedder.transform(img)  # CHW float32 tensor
    x = x.unsqueeze(0).to(embedder.device)

    with torch.inference_mode():
        emb = embedder.model.forward_embedding(x).detach().cpu().numpy().astype(np.float32)[0]
    return emb  # shape: (D,)
