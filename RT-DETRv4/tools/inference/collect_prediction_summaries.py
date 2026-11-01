from pathlib import Path
import csv

ROOT = Path("outputs/rtv4_hgnetv2_s_fsoco")
OUT = ROOT / "conf_speed_comparison.csv"

summary_files = sorted(ROOT.glob("**/prediction_summary.csv"))

if not summary_files:
    raise SystemExit("No prediction_summary.csv files found.")

rows = []
for path in summary_files:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["summary_file"] = str(path)
            rows.append(row)

fieldnames = list(rows[0].keys())

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Found {len(summary_files)} summary files.")
print(f"Wrote {OUT}")
for r in rows:
    print(r)
