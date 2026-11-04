from pathlib import Path
import csv
import json

import cv2
import numpy as np
import pandas as pd

from day07_anomaly_core import (
    anomaly_score,
    create_backbone,
    extract_features,
)


DRIFT_ROOT = Path("samples/day09/drift")
RULE_PATH = Path("configs/day07_anomaly_rules.json")
OUTPUT_PATH = Path("reports/day09/drift_results.csv")
SUMMARY_PATH = Path("reports/day09/drift_summary.csv")


if not RULE_PATH.exists():
    raise FileNotFoundError(RULE_PATH)

rules = json.loads(
    RULE_PATH.read_text(encoding="utf-8")
)

threshold = float(rules["anomaly_threshold"])
reference_path = Path(
    rules.get(
        "reference_path",
        "artifacts/day07/normal_reference.npz",
    )
)

if not reference_path.exists():
    raise FileNotFoundError(reference_path)

reference_data = np.load(reference_path)
reference_vector = reference_data["reference_vector"]

model = create_backbone()
rows = []

for condition in ["baseline", "lighting"]:
    folder = DRIFT_ROOT / condition
    image_files = sorted(folder.glob("*.png"))

    if not image_files:
        raise RuntimeError(
            f"Drift 이미지가 없습니다: {folder}"
        )

    for image_path in image_files:
        gray = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if gray is None:
            raise FileNotFoundError(image_path)

        mean_brightness = float(np.mean(gray))

        feature, _ = extract_features(
            model,
            image_path,
        )

        score = anomaly_score(
            feature,
            reference_vector,
        )

        predicted = (
            "Anomaly"
            if score > threshold
            else "Normal"
        )

        rows.append(
            {
                "file": image_path.name,
                "condition": condition,
                "actual": "Normal",
                "mean_brightness": round(mean_brightness, 3),
                "anomaly_score": round(score, 6),
                "threshold": round(threshold, 6),
                "predicted": predicted,
                "false_positive": predicted == "Anomaly",
            }
        )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)


df = pd.DataFrame(rows)

summary = (
    df.groupby("condition")
    .agg(
        image_count=("file", "count"),
        brightness_mean=("mean_brightness", "mean"),
        anomaly_score_mean=("anomaly_score", "mean"),
        anomaly_score_std=("anomaly_score", "std"),
        false_positive_rate=("false_positive", "mean"),
    )
    .reset_index()
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("=" * 70)
print("DAY09 DRIFT RESULT")
print("=" * 70)
print(summary.to_string(index=False))
print()
print("Threshold:", threshold)
print("Detail   :", OUTPUT_PATH)
print("Summary  :", SUMMARY_PATH)
