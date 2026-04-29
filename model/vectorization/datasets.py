from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    path: Path
    class_id: int
    class_name: str


@dataclass
class SamplesWithIdMapping:
    samples: list[Sample]
    class_to_id: dict[str, int]



class BellyIdDataset(Dataset):
    """
    Expects:
      root/
        1/
          *.jpg
          subfolders/.../*.jpg
        2/
        ...
        21 (error)/   <-- will be skipped by default
    """

    KNOWN_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root: Path,
        transform: nn.Module,
        *,
        min_images_per_class: int = 1,
        samples_with_id_mapping: SamplesWithIdMapping | None = None,
        duplicate_rot180: bool = False,
    ):
        self.root = root
        self.transform = transform
        self.duplicate_rot180 = duplicate_rot180
        self.class_to_id: dict[str, int] | None = None

        if samples_with_id_mapping is not None:
            self.samples = samples_with_id_mapping.samples
            self.class_to_id = samples_with_id_mapping.class_to_id
            return

        # 1) Выбираем папки-классы
        class_dirs: list[Path] = []
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            name = p.name
            class_dirs.append(p)

        if not class_dirs:
            raise RuntimeError(f"No class folders found in {root}")

        # 2) Собираем изображения по классам (рекурсивно)
        class_to_paths: dict[str, list[Path]] = {}
        for class_dir in class_dirs:
            class_name = class_dir.name
            paths = [
                img_path
                for img_path in class_dir.rglob("*")
                if img_path.is_file() and img_path.suffix.lower() in self.KNOWN_IMAGE_EXTENSIONS
            ]
            if len(paths) >= min_images_per_class:
                class_to_paths[class_name] = sorted(paths)

        if not class_to_paths:
            raise RuntimeError(
                f"No images found under {root} (after filtering). "
                f"Try lowering min_images_per_class or check extensions."
            )

        # 3) Маппинг class_name -> class_id (0..C-1)
        class_names = sorted(class_to_paths.keys(), key=lambda s: int(s) if s.isdigit() else s)
        if self.class_to_id is None:
            class_to_id = {name: i for i, name in enumerate(class_names)}
            self.class_to_id = class_to_id

        # 4) Финальный список сэмплов
        samples: list[Sample] = []
        for class_name in class_names:
            cid = self.class_to_id[class_name]
            for img_path in class_to_paths[class_name]:
                samples.append(Sample(path=img_path, class_id=cid, class_name=class_name))

        self.samples = samples

    def __len__(self) -> int:
        if self.duplicate_rot180:
            return len(self.samples) * 2
        return len(self.samples)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        rotated = False
        real_idx = idx

        if self.duplicate_rot180:
            rotated = idx >= len(self.samples)
            real_idx = idx % len(self.samples)

        sample = self.samples[real_idx]
        img = Image.open(sample.path).convert("RGB")

        if rotated:
            img = img.rotate(180)

        x = self.transform(img)
        return x, sample.class_id, sample.path.as_posix()


@dataclass
class EmbedderDatasets:
    train: BellyIdDataset
    val: BellyIdDataset
    class_to_id: dict[str, int]


def build_datasets(
    data_root: Path,
    train_tf: nn.Module,
    val_tf: nn.Module,
    val_ratio: float,
    seed: int | None = None,
) -> EmbedderDatasets:
    # 1. базовый датасет (без аугментаций)
    base_dataset = BellyIdDataset(
        data_root,
        transform=val_tf,
        duplicate_rot180=False,
    )

    if base_dataset.class_to_id is None:
        raise RuntimeError("class_to_id was not initialized")

    class_to_id = base_dataset.class_to_id

    # 2. split
    indices = list(range(len(base_dataset.samples)))
    Random(seed).shuffle(indices)

    val_n = max(1, int(len(indices) * val_ratio))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    train_samples = [base_dataset.samples[i] for i in train_idx]
    val_samples = [base_dataset.samples[i] for i in val_idx]

    # 3. финальные датасеты
    train_dataset = BellyIdDataset(
        data_root,
        transform=train_tf,
        samples_with_id_mapping=SamplesWithIdMapping(
            class_to_id=class_to_id,
            samples=train_samples,
        ),
    )

    val_dataset = BellyIdDataset(
        data_root,
        transform=val_tf,
        samples_with_id_mapping=SamplesWithIdMapping(
            class_to_id=class_to_id,
            samples=val_samples,
        ),
        duplicate_rot180=False,
    )

    return EmbedderDatasets(
        train=train_dataset,
        val=val_dataset,
        class_to_id=class_to_id,
    )


@dataclass
class EvaluationDatasets:
    gallery: BellyIdDataset
    query: BellyIdDataset
    class_to_id: dict[str, int]


def build_evaluation_datasets(
    data_root: Path,
    transform: nn.Module,
    seed: int | None = None,
) -> EvaluationDatasets:
    base_dataset = BellyIdDataset(
        data_root,
        transform=transform,
        min_images_per_class=2,
        duplicate_rot180=False,
    )

    if base_dataset.class_to_id is None:
        raise RuntimeError("class_to_id was not initialized")

    # группируем sample-ы по class_name
    samples_by_class: dict[str, list[Sample]] = {}
    for sample in base_dataset.samples:
        samples_by_class.setdefault(sample.class_name, []).append(sample)

    random = Random(seed)

    gallery_samples: list[Sample] = []
    query_samples: list[Sample] = []

    kept_class_names: list[str] = []

    for class_name in sorted(samples_by_class.keys()):
        class_samples = samples_by_class[class_name][:]

        # перемешиваем изображения внутри особи воспроизводимо
        random.shuffle(class_samples)

        if len(class_samples) < 2:
            continue

        gallery_count = max(1, len(class_samples) // 2)

        gallery_part = class_samples[:gallery_count]
        query_part = class_samples[gallery_count:]

        if not query_part:
            continue

        gallery_samples.extend(gallery_part)
        query_samples.extend(query_part)
        kept_class_names.append(class_name)

    if not kept_class_names:
        raise RuntimeError(
            "No valid classes for retrieval evaluation. "
            "Need at least one class with enough images for both gallery and query."
        )

    # делаем новый компактный mapping только по реально оставшимся классам
    class_to_id = {class_name: idx for idx, class_name in enumerate(kept_class_names)}

    def remap_samples(samples: list[Sample]) -> list[Sample]:
        remapped: list[Sample] = []
        for sample in samples:
            remapped.append(
                Sample(
                    path=sample.path,
                    class_id=class_to_id[sample.class_name],
                    class_name=sample.class_name,
                )
            )
        return remapped

    gallery_samples = remap_samples(gallery_samples)
    query_samples = remap_samples(query_samples)

    gallery_dataset = BellyIdDataset(
        data_root,
        transform=transform,
        samples_with_id_mapping=SamplesWithIdMapping(
            samples=gallery_samples,
            class_to_id=class_to_id,
        ),
        duplicate_rot180=False,
    )

    query_dataset = BellyIdDataset(
        data_root,
        transform=transform,
        samples_with_id_mapping=SamplesWithIdMapping(
            samples=query_samples,
            class_to_id=class_to_id,
        ),
        duplicate_rot180=False,
    )

    return EvaluationDatasets(
        gallery=gallery_dataset,
        query=query_dataset,
        class_to_id=class_to_id,
    )
