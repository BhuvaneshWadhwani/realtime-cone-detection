import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from engine.core import YAMLConfig


CLASS_NAMES = {
    0: "yellow_cone",
    1: "blue_cone",
    2: "orange_cone",
    3: "large_orange_cone",
    4: "unknown_cone",
}


def build_model(config_path, checkpoint_path, device):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(device)
    model.eval()
    return model


def draw_predictions(image, labels, boxes, scores, conf):
    draw = ImageDraw.Draw(image)

    for label, box, score in zip(labels, boxes, scores):
        if float(score) < conf:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        cls_id = int(label)
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        caption = f"{name} {float(score):.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), caption, fill="blue")

    return image


@torch.no_grad()
def predict_one(model, image_path, device, imgsz, conf, output_dir):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    transforms = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.ToTensor(),
    ])

    image_tensor = transforms(image).unsqueeze(0).to(device)
    orig_size = torch.tensor([[w, h]], device=device)

    start = time.perf_counter()
    labels, boxes, scores = model(image_tensor, orig_size)
    elapsed = time.perf_counter() - start

    labels = labels[0].detach().cpu()
    boxes = boxes[0].detach().cpu()
    scores = scores[0].detach().cpu()

    vis = draw_predictions(image.copy(), labels, boxes, scores, conf)
    out_path = output_dir / image_path.name
    vis.save(out_path)

    kept = int((scores >= conf).sum().item())
    return elapsed, kept, str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-i", "--input-dir", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("-d", "--device", default="cuda")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    )

    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    model = build_model(args.config, args.resume, args.device)

    rows = []
    total_time = 0.0

    # Warmup on first image
    if image_paths:
        predict_one(model, image_paths[0], args.device, args.imgsz, args.conf, output_dir)

    for idx, image_path in enumerate(image_paths, start=1):
        elapsed, kept, out_path = predict_one(
            model, image_path, args.device, args.imgsz, args.conf, output_dir
        )
        total_time += elapsed

        rows.append({
            "image": str(image_path),
            "output": out_path,
            "seconds": elapsed,
            "detections": kept,
            "imgsz": args.imgsz,
            "conf": args.conf,
        })

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(image_paths)} images")

    csv_path = output_dir / "prediction_speed.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)

    n = len(image_paths)
    seconds_per_image = total_time / n if n else 0
    fps = n / total_time if total_time > 0 else 0

    summary_path = output_dir / "prediction_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "checkpoint",
                "num_speed_images",
                "imgsz",
                "conf",
                "seconds_per_image",
                "fps",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "run_name": "rtdetrv4_s_fsoco",
            "checkpoint": args.resume,
            "num_speed_images": n,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "seconds_per_image": seconds_per_image,
            "fps": fps,
            "output_dir": str(output_dir),
        })

    print("Done.")
    print(f"Images: {n}")
    print(f"Seconds/image: {seconds_per_image:.6f}")
    print(f"FPS: {fps:.2f}")
    print(f"Saved visualizations to: {output_dir}")
    print(f"Saved speed CSV to: {csv_path}")
    print(f"Saved summary CSV to: {summary_path}")


if __name__ == "__main__":
    main()
