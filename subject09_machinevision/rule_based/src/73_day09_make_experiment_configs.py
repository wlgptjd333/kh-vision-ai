from copy import deepcopy
from pathlib import Path
import json

import pandas as pd


BASE_RULE_PATH = Path("configs/day05_logo_rules.json")
CALIBRATION_PATH = Path(
    "reports/day05_calibration_measurements.csv"
)
OUTPUT_DIR = Path("configs/day09")


if not BASE_RULE_PATH.exists():
    raise FileNotFoundError(BASE_RULE_PATH)

if not CALIBRATION_PATH.exists():
    raise FileNotFoundError(CALIBRATION_PATH)


base_rule = json.loads(
    BASE_RULE_PATH.read_text(encoding="utf-8")
)

required = base_rule.get("required_rules", {})

required_keys = {
    "homography_found",
    "sift_good_matches_min",
    "inlier_ratio_min",
}

missing_keys = required_keys - set(required.keys())

if missing_keys:
    raise KeyError(
        f"Day05 Rule Key가 부족합니다: {sorted(missing_keys)}"
    )


calibration = pd.read_csv(CALIBRATION_PATH)

required_columns = {
    "actual",
    "inlier_ratio",
}

missing_columns = required_columns - set(calibration.columns)

if missing_columns:
    raise RuntimeError(
        "Day05 Calibration CSV 컬럼이 부족합니다: "
        f"{sorted(missing_columns)}"
    )


actual = (
    calibration["actual"]
    .astype(str)
    .str.strip()
    .str.upper()
)

ok_rows = calibration.loc[actual == "OK"].copy()

if ok_rows.empty:
    raise RuntimeError(
        "Calibration에서 actual=OK 행을 찾을 수 없습니다."
    )


baseline_ratio = float(
    required["inlier_ratio_min"]
)

candidate_ratio = round(
    float(
        pd.to_numeric(
            ok_rows["inlier_ratio"],
            errors="raise",
        ).min()
    ),
    6,
)

if candidate_ratio >= baseline_ratio:
    raise RuntimeError(
        "Calibration OK 데이터에서 Baseline보다 낮은 "
        "Inlier Ratio 후보를 얻지 못했습니다. "
        "Test 값을 이용해 Candidate를 임의로 만들지 마세요."
    )


exp001 = {
    "experiment_id": "EXP-001",
    "description": "Day05 frozen baseline",
    "dataset_version": "day05_calibration_v1",
    "source_rule_version": base_rule.get(
        "baseline_version",
        "UNKNOWN",
    ),
    "changed_variable": "none",
    "required_rules": deepcopy(required),
}

exp002 = deepcopy(exp001)
exp002["experiment_id"] = "EXP-002"
exp002["description"] = (
    "Calibration-derived inlier ratio candidate"
)
exp002["changed_variable"] = "inlier_ratio_min"
exp002["old_value"] = baseline_ratio
exp002["new_value"] = candidate_ratio
exp002["candidate_source"] = (
    "minimum inlier_ratio among Day05 Calibration actual=OK"
)
exp002["required_rules"]["inlier_ratio_min"] = (
    candidate_ratio
)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for experiment in [exp001, exp002]:
    output_path = OUTPUT_DIR / (
        experiment["experiment_id"].lower() + ".json"
    )

    output_path.write_text(
        json.dumps(
            experiment,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Saved:", output_path)

print()
print("Baseline Inlier Ratio :", baseline_ratio)
print("Candidate Inlier Ratio:", candidate_ratio)
print(
    "Candidate Source      :",
    "Day05 Calibration actual=OK minimum",
)
print("Changed Variable       : inlier_ratio_min only")
