import csv
import json
from pathlib import Path

from day07_anomaly_core import (
    anomaly_score,
    create_backbone,
    extract_features,
    load_reference,
)


# --------------------------------------------------
# 1. 입력 / 출력 경로
# --------------------------------------------------
VARIANT_DIR = Path(
    "samples/day07/challenge/variants"
)

REFERENCE_PATH = Path(
    "artifacts/day07/normal_reference.npz"
)

RULE_PATH = Path(
    "configs/day07_anomaly_rules.json"
)

OUTPUT_CSV = Path(
    "reports/day07_challenge_results.csv"
)


# --------------------------------------------------
# 2. 필요한 파일 확인
# --------------------------------------------------
required_paths = [
    VARIANT_DIR,
    REFERENCE_PATH,
    RULE_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"필요한 파일 또는 폴더를 확인하세요: {path}"
        )


# --------------------------------------------------
# 3. Freeze한 Rule 불러오기
# --------------------------------------------------
rules = json.loads(
    RULE_PATH.read_text(
        encoding="utf-8"
    )
)

threshold = float(
    rules["anomaly_threshold"]
)


# --------------------------------------------------
# 4. Normal Reference와 모델 준비
# --------------------------------------------------
reference_vector, _ = load_reference(
    REFERENCE_PATH
)

model = create_backbone()

rows = []


# --------------------------------------------------
# 5. 변형 이미지 전체 검사
# --------------------------------------------------
for image_path in sorted(
    VARIANT_DIR.glob("*.png")
):
    vector, _ = extract_features(
        model,
        image_path,
    )

    score = anomaly_score(
        vector,
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
            "actual": "Normal",
            "anomaly_score": round(
                score,
                6,
            ),
            "threshold": round(
                threshold,
                6,
            ),
            "predicted": predicted,
            "result": (
                "PASS"
                if predicted == "Normal"
                else "FP"
            ),
        }
    )


if not rows:
    raise RuntimeError(
        "검사할 Challenge 이미지가 없습니다."
    )


# --------------------------------------------------
# 6. CSV 저장
# --------------------------------------------------
OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "file",
            "actual",
            "anomaly_score",
            "threshold",
            "predicted",
            "result",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


# --------------------------------------------------
# 7. 결과 출력
# --------------------------------------------------
print("=" * 70)
print("DAY07 CHALLENGE")
print("=" * 70)

print(
    "Threshold:",
    threshold,
)

for row in rows:
    print(
        row["file"],
        "score=",
        row["anomaly_score"],
        "predicted=",
        row["predicted"],
        "result=",
        row["result"],
    )

print()

print(
    "Saved:",
    OUTPUT_CSV,
)
