import json
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# === НАСТРОЙКИ ===
DATA_ROOT = Path("dataset")
OUTPUT_ROOT = Path("masks_manual")
OUTPUT_ROOT.mkdir(exist_ok=True)

def process_vgg_json(json_path: Path, img_dir: Path, out_dir: Path):
    with open(json_path, "r") as f:
        vgg_data = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for img_key, img_data in tqdm(vgg_data.items(), desc=f"{json_path.parent.name}"):
        filename = img_data["filename"]
        regions = img_data.get("regions", {})

        if not regions:
            continue

        img_path = img_dir / filename
        if not img_path.exists():
            # пробуем с разными регистрами расширений
            for ext in [".JPG", ".jpg", ".jpeg", ".JPEG", ".png"]:
                alt = img_path.with_suffix(ext)
                if alt.exists():
                    img_path = alt
                    break
            else:
                print(f"⚠️ Пропущен (нет исходного файла): {filename}")
                continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Ошибка чтения {img_path}")
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Support both dict (older VGG) and list (newer)
        if isinstance(regions, dict):
            regions = regions.values()

        for region in regions:
            shape = region.get("shape_attributes", {})
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if not xs or not ys:
                continue
            pts = np.array(list(zip(xs, ys)), dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        if mask.sum() == 0:
            continue  # пустая маска

        out_path = out_dir / f"{Path(filename).stem}_mask.png"
        cv2.imwrite(str(out_path), mask)
        saved += 1

    print(f"✅ {saved} масок сохранено в {out_dir}")


def main():
    for species_dir in DATA_ROOT.iterdir():
        if not species_dir.is_dir():
            continue
        for subdir in sorted(species_dir.iterdir()):
            json_path = subdir / "annotations_vgg.json"
            if not json_path.exists():
                continue

            out_dir = OUTPUT_ROOT / species_dir.name / subdir.name
            print(f"📂 Обрабатываю {json_path.relative_to(DATA_ROOT)}")
            process_vgg_json(json_path, subdir, out_dir)


if __name__ == "__main__":
    main()
