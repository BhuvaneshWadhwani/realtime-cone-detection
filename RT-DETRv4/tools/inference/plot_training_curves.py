import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt

LOG = Path("outputs/rtv4_hgnetv2_s_fsoco/log.txt")
OUT_DIR = Path("outputs/rtv4_hgnetv2_s_fsoco/report_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

epochs = []
map50_95 = []
map50 = []

for line in LOG.read_text(errors="ignore").splitlines():
    line = line.strip()
    if not line:
        continue

    try:
        data = ast.literal_eval(line)
    except Exception:
        try:
            data = json.loads(line)
        except Exception:
            continue

    if "epoch" not in data:
        continue

    epoch = data.get("epoch")
    coco_eval = data.get("test_coco_eval_bbox")

    if coco_eval and isinstance(coco_eval, list) and len(coco_eval) >= 2:
        epochs.append(epoch)
        map50_95.append(coco_eval[0])
        map50.append(coco_eval[1])

if not epochs:
    raise SystemExit("No COCO eval metrics found in log.txt")

plt.figure()
plt.plot(epochs, map50_95, marker="o", label="mAP@0.50:0.95")
plt.plot(epochs, map50, marker="o", label="AP@0.50")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("RT-DETRv4-S validation performance on FSOCO")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "map_curve.png", dpi=200)

print(f"Saved: {OUT_DIR / 'map_curve.png'}")
print(f"Best mAP@0.50:0.95: {max(map50_95):.4f} at epoch {epochs[map50_95.index(max(map50_95))]}")
