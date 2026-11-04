from pathlib import Path

import pandas as pd


INPUT_PATH = Path("reports/day09/experiment_log.csv")
OUTPUT_PATH = Path("reports/day09/experiment_comparison.csv")


df = pd.read_csv(INPUT_PATH)

columns = [
    "experiment_id",
    "changed_variable",
    "old_value",
    "new_value",
    "homography_required",
    "sift_good_matches_min",
    "inlier_ratio_min",
    "samples",
    "accuracy",
    "ng_precision",
    "ng_recall",
    "fp",
    "fn",
]

missing = [column for column in columns if column not in df.columns]

if missing:
    raise RuntimeError(
        f"Experiment Log 컬럼이 부족합니다: {missing}"
    )

comparison = df[columns].copy()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("=" * 70)
print("DAY09 EXPERIMENT COMPARISON")
print("=" * 70)
print(comparison.to_string(index=False))
print()
print("Saved:", OUTPUT_PATH)
