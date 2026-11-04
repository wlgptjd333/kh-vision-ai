# ==================================================
# ① 필요한 라이브러리와 공통 함수 불러오기
#    → csv   : Calibration 결과 CSV 저장
#    → json  : 확정한 판정 규칙 JSON 저장
#    → Path  : 파일과 폴더 경로 관리
#    → NumPy : 5 Percentile Threshold 계산
#
#    day08_feature_core에서 다음 함수를 사용합니다.
#    → create_feature_model() : ResNet18 준비
#    → extract_feature()      : 이미지 특징 추출
#    → cosine_similarity()    : 두 특징의 유사도 계산
# ==================================================
import csv
import json
from pathlib import Path

import numpy as np

from day08_feature_core import (
    create_feature_model,
    cosine_similarity,
    extract_feature,
)


# ==================================================
# ② 입력 / 출력 경로 및 Percentile 설정
#
#    Reference  : 정상 기준 이미지 1장
#    Calibration: Threshold 계산용 정상 이미지 20장
#
#    출력 파일:
#    → 정상 기준 Feature Vector (NPY)
#    → 판정 규칙 (JSON)
#    → Calibration 유사도 기록 (CSV)
# ==================================================

# ----------------------------------------------
# ②-1. One-shot 정상 Reference 이미지 경로
# ----------------------------------------------
REFERENCE_DIR = Path(
    "samples/day08/one_shot/reference"
)

# ----------------------------------------------
# ②-2. 정상 Calibration 이미지 경로
# ----------------------------------------------
CALIBRATION_DIR = Path(
    "samples/day08/one_shot/calibration/normal"
)

# ----------------------------------------------
# ②-3. 정상 기준 Feature Vector 저장 경로
# ----------------------------------------------
ARTIFACT_PATH = Path(
    "artifacts/day08/one_shot_reference.npy"
)

# ----------------------------------------------
# ②-4. 확정한 판정 규칙 JSON 저장 경로
# ----------------------------------------------
RULE_PATH = Path(
    "configs/day08_one_shot_rules.json"
)

# ----------------------------------------------
# ②-5. 이미지별 유사도 CSV 저장 경로
# ----------------------------------------------
CSV_PATH = Path(
    "reports/day08_one_shot_calibration.csv"
)

# ----------------------------------------------
# ②-6. Threshold 기준 설정
#
#       유사도가 낮을수록 정상 기준과 다름
#       → 정상 유사도 분포의 하위 5% 경계값 사용
# ----------------------------------------------
PERCENTILE = 5


# ==================================================
# ③ Reference / Calibration 이미지 확인
#    → Reference가 정확히 1장인지 확인
#    → Calibration 이미지가 있는지 확인
#    → 필요한 데이터가 없으면 오류 발생
# ==================================================

# ----------------------------------------------
# ③-1. Reference 이미지 목록 가져오기
# ----------------------------------------------
reference_files = sorted(
    REFERENCE_DIR.glob("*.png")
)

# ----------------------------------------------
# ③-2. Reference가 정확히 1장인지 확인
#
#       One-shot이므로 기준 이미지는 1장입니다.
# ----------------------------------------------
if len(reference_files) != 1:
    raise RuntimeError(
        f"Reference는 1장이어야 합니다. 현재: {len(reference_files)}"
    )

# ----------------------------------------------
# ③-3. 정상 Reference 이미지 선택
# ----------------------------------------------
reference_path = reference_files[0]

# ----------------------------------------------
# ③-4. Calibration 정상 이미지 목록 가져오기
# ----------------------------------------------
calibration_files = sorted(
    CALIBRATION_DIR.glob("*.png")
)

# ----------------------------------------------
# ③-5. Calibration 이미지 존재 여부 확인
#
#       63번에서 정상 20장을 준비했습니다.
# ----------------------------------------------
if not calibration_files:
    raise RuntimeError(
        "Calibration 이미지가 없습니다."
    )


# ==================================================
# ④ ResNet18 모델과 정상 기준 Feature 준비
#
#    → 사전학습된 ResNet18 특징 추출기 생성
#    → Reference 이미지 1장에서 특징 추출
#    → 512차원 Feature Vector를 NPY로 저장
#
#    이 과정은 새로운 모델을 학습하지 않습니다.
# ==================================================

# ----------------------------------------------
# ④-1. ResNet18 특징 추출 모델 생성
# ----------------------------------------------
model = create_feature_model()

# ----------------------------------------------
# ④-2. Reference 이미지의 특징 추출
#
#       정상 이미지 1장
#           ↓
#       ResNet18
#           ↓
#       Feature Vector (512,)
# ----------------------------------------------
reference_feature = extract_feature(
    model,
    reference_path
)

# ----------------------------------------------
# ④-3. 정상 기준 저장 폴더 생성
# ----------------------------------------------
ARTIFACT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------
# ④-4. 정상 기준 Feature Vector 저장
#
#       이후 65번 검사에서 재사용합니다.
# ----------------------------------------------
np.save(
    ARTIFACT_PATH,
    reference_feature
)


# ==================================================
# ⑤ Calibration 이미지의 Cosine Similarity 계산
#
#    → 정상 Calibration 이미지를 한 장씩 읽기
#    → 이미지마다 512차원 Feature 추출
#    → Reference 특징과 유사도 계산
#    → 계산 결과를 리스트에 저장
#
#    정상 이미지라도 유사도는 서로 다를 수 있습니다.
# ==================================================

# 이미지별 파일명과 유사도를 저장할 리스트
rows = []

# Threshold 계산에 사용할 유사도 리스트
similarities = []

# ----------------------------------------------
# ⑤-1. 정상 Calibration 이미지 반복
# ----------------------------------------------
for image_path in calibration_files:

    # ------------------------------------------
    # ⑤-2. 현재 이미지에서 Feature 추출
    # ------------------------------------------
    feature = extract_feature(
        model,
        image_path
    )

    # ------------------------------------------
    # ⑤-3. Reference와 코사인 유사도 계산
    #
    #       유사도 높음 → 정상 기준과 비슷함
    #       유사도 낮음 → 정상 기준과 다름
    # ------------------------------------------
    similarity = cosine_similarity(
        reference_feature,
        feature,
    )

    # ------------------------------------------
    # ⑤-4. Threshold 계산용 유사도 저장
    # ------------------------------------------
    similarities.append(
        similarity
    )

    # ------------------------------------------
    # ⑤-5. 파일명과 유사도를 CSV용으로 기록
    #       → 표시할 값은 소수점 6자리
    # ------------------------------------------
    rows.append(
        {
            "file": image_path.name,
            "similarity": round(
                similarity,
                6
            ),
        }
    )


# ==================================================
# ⑥ 5 Percentile Threshold 계산
#
#    → 정상 Calibration 유사도들의 분포 사용
#    → 하위 5% 경계에 해당하는 값 계산
#
#    이 기준보다 유사도가 낮으면 Anomaly
#    이 기준 이상이면 Normal로 판정합니다.
#
#    주의!
#    Threshold는 실제 Calibration 결과로 계산하며,
#    여기서 임의의 숫자를 지정하지 않습니다.
# ==================================================
threshold = float(
    np.percentile(
        similarities,
        PERCENTILE,
    )
)


# ==================================================
# ⑦ 이미지별 Calibration 결과 CSV 저장
#
#    → 파일명과 코사인 유사도를 기록
#    → 이후 정상 유사도 분포를 확인할 수 있음
# ==================================================

# ----------------------------------------------
# ⑦-1. reports 폴더가 없으면 생성
# ----------------------------------------------
CSV_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------
# ⑦-2. CSV 파일 열기
# ----------------------------------------------
with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    # ------------------------------------------
    # ⑦-3. CSV 컬럼명 설정
    # ------------------------------------------
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "file",
            "similarity",
        ],
    )

    # ------------------------------------------
    # ⑦-4. CSV 컬럼명 작성
    # ------------------------------------------
    writer.writeheader()

    # ------------------------------------------
    # ⑦-5. 이미지별 유사도 기록
    # ------------------------------------------
    writer.writerows(
        rows
    )


# ==================================================
# ⑧ One-shot 판정 규칙 구성 및 JSON 저장
#
#    → 기준 이미지 정보 기록
#    → 모델과 검사 방법 기록
#    → Calibration 이미지 수 기록
#    → Threshold와 판정 규칙 기록
#
#    Rule Freeze:
#    → Query 검사 전에 판정 기준을 저장하여 고정
# ==================================================

# ----------------------------------------------
# ⑧-1. One-shot Rule 정보 구성
# ----------------------------------------------
rules = {

    # One-shot Baseline 버전
    "baseline_version": "day08-one-shot-v1.0",

    # 사용 데이터셋
    "dataset": "MVTec AD bottle",

    # 검사 방법
    "method": "one_reference_cosine_similarity",

    # 특징 추출 모델
    "backbone": "resnet18_imagenet_pretrained",

    # 정상 Reference 이미지 수
    "reference_count": 1,

    # Reference 이미지 파일명
    "reference_file": reference_path.name,

    # 저장된 Reference Feature 경로
    "reference_path": str(ARTIFACT_PATH),

    # Calibration 이미지 개수
    "calibration_count": len(calibration_files),

    # Threshold 계산 방법
    "threshold_method": "normal_calibration_5_percentile",

    # 실제 계산된 유사도 Threshold
    "similarity_threshold": threshold,

    # 정상 / 이상 판정 규칙
    "decision_rule": "similarity >= threshold => Normal",
}

# ----------------------------------------------
# ⑧-2. configs 폴더가 없으면 생성
# ----------------------------------------------
RULE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------
# ⑧-3. Rule을 JSON 파일로 저장
#
#       이후 65번 검사에서 그대로 사용합니다.
# ----------------------------------------------
RULE_PATH.write_text(
    json.dumps(
        rules,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ==================================================
# ⑨ 실행 결과 확인
#
#    → Reference 파일명 출력
#    → Calibration 이미지 개수 출력
#    → 유사도 최솟값 / 평균 / 최댓값 출력
#    → Threshold 및 판정 규칙 출력
#    → 저장한 Rule JSON 경로 출력
#
#    아직 Common Query 검사는 하지 않습니다.
# ==================================================
print("=" * 70)
print("DAY08 ONE-SHOT CALIBRATION")
print("=" * 70)

# ----------------------------------------------
# ⑨-1. 사용한 Reference 파일명
# ----------------------------------------------
print(
    "Reference          :",
    reference_path.name
)

# ----------------------------------------------
# ⑨-2. Calibration 이미지 개수
# ----------------------------------------------
print(
    "Calibration Images :",
    len(calibration_files)
)

# ----------------------------------------------
# ⑨-3. Calibration 유사도 최솟값
# ----------------------------------------------
print(
    "Similarity Min     :",
    round(min(similarities), 6)
)

# ----------------------------------------------
# ⑨-4. Calibration 유사도 평균
# ----------------------------------------------
print(
    "Similarity Mean    :",
    round(float(np.mean(similarities)), 6)
)

# ----------------------------------------------
# ⑨-5. Calibration 유사도 최댓값
# ----------------------------------------------
print(
    "Similarity Max     :",
    round(max(similarities), 6)
)

# ----------------------------------------------
# ⑨-6. 5 Percentile Threshold 출력
# ----------------------------------------------
print(
    "5 Percentile       :",
    round(threshold, 6)
)

# ----------------------------------------------
# ⑨-7. 확정된 판정 규칙 출력
# ----------------------------------------------
print(
    "Rule               : similarity >= threshold => Normal"
)

# ----------------------------------------------
# ⑨-8. Rule JSON 저장 경로 출력
# ----------------------------------------------
print(
    "Saved Rule         :",
    RULE_PATH
)
