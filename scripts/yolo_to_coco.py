from pathlib import Path
import json
import shutil
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

YOLO_DIR = PROJECT_ROOT / "data" / "fsoco_yolo"
COCO_DIR = PROJECT_ROOT / "data" / "fsoco_coco"

IMG_IN = YOLO_DIR / "images"
LBL_IN = YOLO_DIR / "labels"

IMG_OUT = COCO_DIR / "images"
ANN_OUT = COCO_DIR / "annotations"

SPLITS = ["train", "val", "test"]


def load_class_names():
    yaml_path = YOLO_DIR / "data.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]

    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    return names


def convert_split(split, class_names):
    image_dir = IMG_IN / split
    label_dir = LBL_IN / split

    output_image_dir = IMG_OUT / split
    output_image_dir.mkdir(parents=True, exist_ok=True)
    ANN_OUT.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []

    ann_id = 1
    img_id = 1

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_paths.extend(image_dir.glob(ext))

    image_paths = sorted(image_paths)

    for image_path in image_paths:
        shutil.copy2(image_path, output_image_dir / image_path.name)

        with Image.open(image_path) as img:
            width, height = img.size

        images.append({
            "id": img_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            lines = label_path.read_text(encoding="utf-8").strip().splitlines()

            for line in lines:
                if not line.strip():
                    continue

                cls_id, x_center, y_center, box_w, box_h = map(float, line.split())

                x_min = (x_center - box_w / 2) * width
                y_min = (y_center - box_h / 2) * height
                abs_w = box_w * width
                abs_h = box_h * height

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls_id) + 1,
                    "bbox": [x_min, y_min, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0,
                })

                ann_id += 1

        img_id += 1

    categories = [
        {
            "id": i + 1,
            "name": name,
            "supercategory": "cone",
        }
        for i, name in enumerate(class_names)
    ]

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_json = ANN_OUT / f"instances_{split}.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"Saved {output_json}")
    print(f"{split}: {len(images)} images, {len(annotations)} annotations")


def main():
    class_names = load_class_names()
    print("Classes:", class_names)

    for split in SPLITS:
        convert_split(split, class_names)


if __name__ == "__main__":
    main()