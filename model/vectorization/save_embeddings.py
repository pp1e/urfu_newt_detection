from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Dict

import psycopg
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent.parent
    )
)

from model.vectorization.common import build_transforms
from model.vectorization.common import ResNetClassifierEmbedder
from tg_bot.settings.database_config import SYNC_DATABASE_URL


@dataclass(frozen=True)
class Sample:
    path: Path
    class_name: str


class BellyFolderDataset(Dataset):
    """
    Expects:
      root/
        1/ ... images (recursive)
        2/ ... images
        ...
    Non-numeric folders are skipped by default.
    """
    def __init__(self, root: Path, transform: nn.Module, allow_non_numeric: bool = False):
        self.root = root
        self.transform = transform

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        numeric_re = re.compile(r"^\d+$")

        samples: List[Sample] = []
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            name = class_dir.name
            if not allow_non_numeric and not numeric_re.match(name):
                continue

            for p in sorted(class_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts:
                    samples.append(Sample(path=p, class_name=name))

        if not samples:
            raise RuntimeError(f"No images found under {root}")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.transform(img)
        rel_path = str(s.path.relative_to(self.root))
        return x, s.class_name, rel_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Embed belly images and insert into Postgres")
    p.add_argument("--data-root", type=Path, required=True, help="Root folder with class subfolders")
    p.add_argument("--checkpoint", type=Path, required=True, help="Trained embedder .pt")
    p.add_argument("--run-name", type=str, required=True, help="Run/model name")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def to_vector_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def iter_items(obj: Any) -> Iterable[dict]:
    if isinstance(obj, dict) and "items" in obj:
        items = obj["items"]
    else:
        items = obj

    if not isinstance(items, (list, tuple)):
        raise TypeError("Embeddings file must contain list/tuple of dicts or dict with key 'items'.")

    for it in items:
        if not isinstance(it, dict):
            raise TypeError("Each item must be a dict.")
        yield it



def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    class_to_id: Dict[str, int] = ckpt.get("class_to_id")
    if not isinstance(class_to_id, dict) or not class_to_id:
        raise RuntimeError("Checkpoint doesn't contain 'class_to_id'. Use the training checkpoint you saved.")

    embedding_dim = int(ckpt.get("embedding_dim", 256))
    image_size = int(ckpt.get("image_size", 224))
    num_classes = len(class_to_id)

    _, val_tf = build_transforms(image_size)

    dataset = BellyFolderDataset(
        root=args.data_root,
        transform=val_tf,
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device(args.device)

    classifier = ResNetClassifierEmbedder(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=False,
    )
    classifier.load_state_dict(ckpt["model_state"])
    classifier.to(device)
    classifier.eval()

    insert_sql = """
        INSERT INTO belly_embedding (
            run_name,
            newt_class_name,
            image_sha256,
            image_bytes,
            embedding
        )
        VALUES (%s, %s, %s, %s, %s::vector)
        ON CONFLICT (image_sha256)
        DO UPDATE SET
            run_name = EXCLUDED.run_name,
            newt_class_name = EXCLUDED.newt_class_name,
            image_bytes = EXCLUDED.image_bytes,
            embedding = EXCLUDED.embedding;
    """

    attempted = 0

    # ---- DB connect
    with psycopg.connect(SYNC_DATABASE_URL) as conn:
        batch_rows: List[Tuple[str, str, str, bytes, str]] = []

        def flush() -> None:
            nonlocal attempted, batch_rows
            if not batch_rows:
                return
            with conn.cursor() as cur:
                cur.executemany(insert_sql, batch_rows)
            conn.commit()
            attempted += len(batch_rows)
            batch_rows = []

        # ---- Inference + insert
        with torch.inference_mode():
            for x, class_names, rel_paths in tqdm(loader, desc="Embedding+Insert", unit="batch"):
                x = x.to(device, non_blocking=True)
                emb = classifier.forward_embedding(x).detach().cpu().numpy().astype(np.float32)  # (B, D)

                for i in range(emb.shape[0]):
                    vec_literal = to_vector_literal(emb[i])

                    img_path = args.data_root / rel_paths[i]
                    image_bytes = img_path.read_bytes()
                    image_sha256 = hashlib.sha256(image_bytes).hexdigest()

                    batch_rows.append(
                        (
                            args.run_name,
                            str(class_names[i]),
                            image_sha256,
                            image_bytes,
                            vec_literal,
                        )
                    )

                if len(batch_rows) >= 500:
                    flush()

        flush()

    print(f"Done. attempted_inserts={attempted}.")


if __name__ == "__main__":
    main()
