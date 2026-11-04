from pathlib import Path
import csv
import json
import subprocess


# --------------------------------------------------
# 1. 전체 학습 이력 확인용
#    없어도 오늘 계산을 즉시 중단하지는 않지만
#    이전 결과가 어디까지 남아 있는지 확인합니다.
# --------------------------------------------------
HISTORY_FILES = [
    Path("reports/day01_problem_definition.md"),
    Path("reports/day02_parameter_table.csv"),
    Path("reports/day03_inspection_results.csv"),
    Path("reports/day04_homography_results.csv"),
]


# --------------------------------------------------
# 2. Day09 핵심 실습에 필요한 필수 파일
# --------------------------------------------------
REQUIRED_FILES = [
    # Day05
    Path("configs/day05_logo_rules.json"),
    Path("reports/day05_calibration_measurements.csv"),

    # Day06
    Path("configs/day06_ocr_rules.json"),
    Path("configs/day06_labels.csv"),
    Path("reports/day06_ocr_results.csv"),

    # Day07
    Path("configs/day07_anomaly_rules.json"),
    Path("artifacts/day07/normal_reference.npz"),
    Path("reports/day07_anomaly_scores.csv"),

    # Day08 CORE
    Path("configs/day08_one_shot_rules.json"),
    Path("configs/day08_fewshot_config.json"),
    Path("artifacts/day08/one_shot_reference.npy"),
    Path("artifacts/day08/fewshot_prototypes.npz"),
    Path("reports/day08_one_shot_results.csv"),
    Path("reports/day08_fewshot_results.csv"),
    Path("reports/day08_method_comparison.csv"),
]


REQUIRED_DIRS = [
    Path("samples/day07/test/normal"),
    Path("samples/day08/common_query/normal"),
    Path("samples/day08/common_query/anomaly/broken_small"),
]


DAY05_REQUIRED_COLUMNS = {
    "file",
    "actual",
    "template_score",
    "sift_good_matches",
    "homography_found",
    "homography_inliers",
    "inlier_ratio",
}


DAY05_REQUIRED_RULE_KEYS = {
    "homography_found",
    "sift_good_matches_min",
    "inlier_ratio_min",
}


DAY06_REQUIRED_COLUMNS = {
    "file",
    "actual_text",
    "expected_text",
    "actual_status",
    "final_text",
    "format_ok",
    "expected_match",
    "predicted_status",
    "reasons",
}


DAY08_RESULT_COLUMNS = {
    "sample_id",
    "file",
    "actual",
    "predicted",
    "correct",
}


DAY08_COMPARISON_COLUMNS = {
    "method",
    "samples",
    "accuracy",
    "anomaly_precision",
    "anomaly_recall",
    "tp",
    "tn",
    "fp",
    "fn",
}


def read_csv_columns(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)
        return set(reader.fieldnames or [])


print("=" * 70)
print("DAY09 ENVIRONMENT CHECK")
print("=" * 70)


# --------------------------------------------------
# 3. Day01~04 학습 이력 확인
# --------------------------------------------------
print("\n[Day01~04 History]")

for path in HISTORY_FILES:
    status = "OK" if path.exists() else "HISTORY MISSING"
    print(f"[{status}] {path}")


# --------------------------------------------------
# 4. Day05~08 필수 파일 확인
# --------------------------------------------------
missing_required = []

print("\n[Required Files]")

for path in REQUIRED_FILES:
    if path.exists():
        print("[OK]     ", path)
    else:
        print("[MISSING]", path)
        missing_required.append(path)


print("\n[Required Directories]")

for path in REQUIRED_DIRS:
    file_count = (
        len(
            [
                p
                for p in path.rglob("*")
                if p.is_file()
            ]
        )
        if path.exists()
        else 0
    )

    if path.exists() and file_count > 0:
        print(
            f"[OK]      {path}  files={file_count}"
        )
    else:
        print(f"[MISSING] {path}")
        missing_required.append(path)


if missing_required:
    print("\n필수 파일 또는 폴더가 부족합니다.")

    for path in missing_required:
        print("-", path)

    raise RuntimeError(
        "Day05~Day08 CORE 산출물을 먼저 확인하세요."
    )


# --------------------------------------------------
# 5. Day05 Calibration CSV Schema
# --------------------------------------------------
day05_columns = read_csv_columns(
    Path(
        "reports/day05_calibration_measurements.csv"
    )
)

missing = DAY05_REQUIRED_COLUMNS - day05_columns

if missing:
    raise RuntimeError(
        "Day05 Calibration CSV 컬럼이 부족합니다: "
        f"{sorted(missing)}"
    )

print("\n[OK] Day05 Calibration CSV Schema")


# --------------------------------------------------
# 6. Day05 Frozen Rule Schema
# --------------------------------------------------
day05_rule = json.loads(
    Path(
        "configs/day05_logo_rules.json"
    ).read_text(
        encoding="utf-8"
    )
)

required_rules = day05_rule.get(
    "required_rules",
    {},
)

missing = (
    DAY05_REQUIRED_RULE_KEYS
    - set(required_rules.keys())
)

if missing:
    raise RuntimeError(
        "Day05 Rule Key가 부족합니다: "
        f"{sorted(missing)}"
    )

print("[OK] Day05 Frozen Rule Schema")


# --------------------------------------------------
# 7. Day06 OCR Result Schema
# --------------------------------------------------
day06_columns = read_csv_columns(
    Path("reports/day06_ocr_results.csv")
)

missing = DAY06_REQUIRED_COLUMNS - day06_columns

if missing:
    raise RuntimeError(
        "Day06 OCR Result 컬럼이 부족합니다: "
        f"{sorted(missing)}"
    )

print("[OK] Day06 OCR Result Schema")


# --------------------------------------------------
# 8. Day07 Frozen Rule + Reference
# --------------------------------------------------
day07_rule = json.loads(
    Path(
        "configs/day07_anomaly_rules.json"
    ).read_text(
        encoding="utf-8"
    )
)

if "anomaly_threshold" not in day07_rule:
    raise KeyError(
        "Day07 Rule에 anomaly_threshold가 없습니다."
    )

reference_path = Path(
    day07_rule.get(
        "reference_path",
        "artifacts/day07/normal_reference.npz",
    )
)

if not reference_path.exists():
    raise FileNotFoundError(
        reference_path
    )

print("[OK] Day07 Frozen Rule / Reference")


# --------------------------------------------------
# 9. Day08 One-shot / Few-shot Result Schema
# --------------------------------------------------
for result_path in [
    Path("reports/day08_one_shot_results.csv"),
    Path("reports/day08_fewshot_results.csv"),
]:
    columns = read_csv_columns(
        result_path
    )

    missing = DAY08_RESULT_COLUMNS - columns

    if missing:
        raise RuntimeError(
            f"{result_path} 컬럼이 부족합니다: "
            f"{sorted(missing)}"
        )

print("[OK] Day08 One/Few-shot Result Schema")


comparison_columns = read_csv_columns(
    Path(
        "reports/day08_method_comparison.csv"
    )
)

missing = (
    DAY08_COMPARISON_COLUMNS
    - comparison_columns
)

if missing:
    raise RuntimeError(
        "Day08 Method Comparison 컬럼이 부족합니다: "
        f"{sorted(missing)}"
    )

print("[OK] Day08 Method Comparison Schema")


# --------------------------------------------------
# 10. Day08 Common Query 기본 수량 확인
# --------------------------------------------------
normal_count = len(
    list(
        Path(
            "samples/day08/common_query/normal"
        ).glob("*.png")
    )
)

anomaly_count = len(
    list(
        Path(
            "samples/day08/common_query/"
            "anomaly/broken_small"
        ).glob("*.png")
    )
)

print()
print("Day08 Common Query")
print("Normal :", normal_count)
print("Anomaly:", anomaly_count)

if normal_count != 10 or anomaly_count != 10:
    raise RuntimeError(
        "Day08 Common Query는 현재 수업 기준 "
        "Normal 10장 + Anomaly 10장을 예상합니다."
    )


# --------------------------------------------------
# 11. Git Commit 확인
# --------------------------------------------------
try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()

except Exception:
    commit = "NO_GIT"


print()
print("Git Commit:", commit)
print("\nDay09 Environment OK")
