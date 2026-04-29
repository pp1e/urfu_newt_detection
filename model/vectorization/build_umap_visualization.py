import argparse
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from model.vectorization.common import save_plot_umap, extract_embeddings, build_model, seed_all
from model.vectorization.datasets import BellyIdDataset
from tg_bot.bot.belly_vectorization.constants import IMAGE_SIZE
from tg_bot.bot.belly_vectorization.build_transform import build_transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-path", type=Path, default=Path("umap.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_size = checkpoint.get("image_size", IMAGE_SIZE)

    _, val_tf = build_transforms(image_size)

    dataset = BellyIdDataset(
        root=args.data_root,
        transform=val_tf,
        min_images_per_class=2,
        duplicate_rot180=False,
    )

    model, _ = build_model(args.checkpoint, device)

    embeddings, labels = extract_embeddings(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    save_plot_umap(
        embeddings=embeddings,
        labels=labels,
        save_path=Path(args.save_path),
        max_classes=25,
    )



if __name__ == "__main__":
    main()
