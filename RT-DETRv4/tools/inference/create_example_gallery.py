from pathlib import Path
import shutil

SRC = Path("outputs/rtv4_hgnetv2_s_fsoco/predictions_test_conf040")
DST = Path("outputs/rtv4_hgnetv2_s_fsoco/report_examples")

DST.mkdir(exist_ok=True)

images = sorted(SRC.glob("*.jpg"))

# Take evenly spaced examples
N = 12
step = max(1, len(images) // N)

selected = images[::step][:N]

for img in selected:
    shutil.copy2(img, DST / img.name)

print(f"Copied {len(selected)} images.")
print(f"Output folder: {DST}")
