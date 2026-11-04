from datetime import datetime
from pathlib import Path
import csv
import json
import subprocess

import pandas as pd


DATA_PATH = Path("reports/day05_calibration_measurements.csv")
CONFIG_DIR = Path("configs/day09")
OUTPUT_PATH = Path("reports/day09/experiment_log.csv")


if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)


def parse_bool(value):
    text = str(value).strip().lower()

    if text in {"1", "true", "yes"}:
        return True

    if text in {"0", "false", "no"}:
        return False

    raise ValueError(
        f"Boolean으로 해석할 수 없는 값입니다: {value}"
    )


def row_passes_rule(row, config):
    required = config["required_rules"]

    homography_ok = True

    if required.get("homography_found", True):
        homography_ok = parse_bool(
            row["homography_found"]
        )

    sift_ok = (
        float(row["sift_good_matches"])
        >= float(required["sift_good_matches_min"])
    )

    inlier_ok = (
        float(row["inlier_ratio"])
        >= float(required["inlier_ratio_min"])
    )

    return all(
        [
            homography_ok,
            sift_ok,
            inlier_ok,
        ]
    )


try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    git_commit = "NO_GIT"


df = pd.read_csv(DATA_PATH)

required_columns = {
    "file",
    "actual",
    "template_score",
    "sift_good_matches",
    "homography_found",
    "homography_inliers",
    "inlier_ratio",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise RuntimeError(
        "Day05 Calibration CSV 컬럼이 부족합니다: "
        f"{sorted(missing_columns)}"
    )

rows = []

for config_path in sorted(CONFIG_DIR.glob("exp-*.json")):
    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    tp = tn = fp = fn = 0

    for _, item in df.iterrows():
        actual = str(item["actual"]).strip().upper()
        predicted = "OK" if row_passes_rule(item, config) else "NG"

        if actual == "NG" and predicted == "NG":
            tp += 1
        elif actual == "OK" and predicted == "OK":
            tn += 1
        elif actual == "OK" and predicted == "NG":
            fp += 1
        elif actual == "NG" and predicted == "OK":
            fn += 1
        else:
            raise RuntimeError(
                f"알 수 없는 actual 값입니다: {actual}"
            )

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    rows.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "experiment_id": config["experiment_id"],
            "dataset_version": config["dataset_version"],
            "code_commit": git_commit,
            "changed_variable": config.get("changed_variable", ""),
            "old_value": config.get("old_value", ""),
            "new_value": config.get("new_value", ""),
            "homography_required": config[
                "required_rules"
            ]["homography_found"],
            "sift_good_matches_min": config[
                "required_rules"
            ]["sift_good_matches_min"],
            "inlier_ratio_min": config[
                "required_rules"
            ]["inlier_ratio_min"],
            "samples": total,
            "accuracy": round(accuracy, 6),
            "ng_precision": round(precision, 6),
            "ng_recall": round(recall, 6),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
    )

if not rows:
    raise RuntimeError("Experiment Config가 없습니다.")

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

print("=" * 70)
print("DAY09 EXPERIMENT RUN")
print("=" * 70)

for row in rows:
    print()
    print("Experiment:", row["experiment_id"])
    print("Changed   :", row["changed_variable"])
    print("Accuracy  :", row["accuracy"])
    print("Precision :", row["ng_precision"])
    print("Recall    :", row["ng_recall"])
    print(
        "TP/TN/FP/FN:",
        row["tp"],
        row["tn"],
        row["fp"],
        row["fn"],
    )

print()
print("Saved:", OUTPUT_PATH)
