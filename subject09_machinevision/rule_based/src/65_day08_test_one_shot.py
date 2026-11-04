# ==================================================
# ① 필요한 라이브러리와 공통 함수 불러오기
#    → csv   : 이미지별 검사 결과 CSV 저장
#    → json  : 64번에서 확정한 판정 규칙 읽기
#    → Path  : 파일과 폴더 경로 관리
#    → NumPy : 저장된 Reference Feature 불러오기
#
#    day08_feature_core의 공통 함수 사용
#    → create_feature_model() : ResNet18 준비
#    → extract_feature()      : 이미지 특징 추출
#    → cosine_similarity()    : 유사도 계산
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
# ② 입력 및 출력 경로 설정
#
#    → 63번에서 준비한 Common Query 사용
#    → 64번에서 저장한 Reference와 Rule 사용
#    → 검사 결과를 CSV 파일로 저장
# ==================================================

# ----------------------------------------------
# ②-1. 검사할 Common Query 폴더
#       → 정상 10장 + 작은 파손 10장
# ----------------------------------------------
QUERY_ROOT = Path(
    "samples/day08/common_query"
)


# ----------------------------------------------
# ②-2. 저장된 정상 Reference Feature
#       → 64번에서 037.png의 특징을 저장한 파일
# ----------------------------------------------
REFERENCE_PATH = Path(
    "artifacts/day08/one_shot_reference.npy"
)


# ----------------------------------------------
# ②-3. 확정된 One-shot 판정 규칙 JSON
#       → Similarity Threshold가 저장되어 있음
# ----------------------------------------------
RULE_PATH = Path(
    "configs/day08_one_shot_rules.json"
)


# ----------------------------------------------
# ②-4. 검사 결과 CSV 저장 경로
# ----------------------------------------------
CSV_PATH = Path(
    "reports/day08_one_shot_results.csv"
)


# ==================================================
# ③ 64번에서 확정한 One-shot 검사 기준 불러오기
#
#    → Reference Feature를 NPY에서 읽기
#    → Rule JSON에서 Threshold 가져오기
#
#    중요한 점!
#    이번 코드에서 Threshold를 새로 계산하지 않고,
#    저장된 값을 그대로 사용합니다.
# ==================================================

# ----------------------------------------------
# ③-1. 정상 Reference Feature 불러오기
#
#       Reference 037.png에서 추출한
#       512차원 Feature Vector입니다.
# ----------------------------------------------
reference_feature = np.load(
    REFERENCE_PATH
)


# ----------------------------------------------
# ③-2. JSON에 저장된 판정 규칙 불러오기
# ----------------------------------------------
rules = json.loads(
    RULE_PATH.read_text(
        encoding="utf-8"
    )
)


# ----------------------------------------------
# ③-3. Similarity Threshold 가져오기
#
#       앞선 실습 결과:
#       Threshold 약 0.982113
#
#       실제 판정에는 JSON의 정밀한 값을 사용
# ----------------------------------------------
threshold = float(
    rules["similarity_threshold"]
)


# ==================================================
# ④ Feature 추출용 ResNet18과 결과 리스트 준비
#
#    → 사전학습된 ResNet18 사용
#    → 새로운 모델 학습은 하지 않음
#    → 검사 결과를 저장할 빈 리스트 생성
# ==================================================

# ----------------------------------------------
# ④-1. ResNet18 특징 추출 모델 준비
# ----------------------------------------------
model = create_feature_model()


# ----------------------------------------------
# ④-2. 이미지별 검사 결과를 저장할 리스트
# ----------------------------------------------
rows = []


# ==================================================
# ⑤ 이미지 한 장을 검사하는 함수 작성
#
#    함수: inspect()
#
#    → 이미지의 Feature 추출
#    → Reference와 유사도 계산
#    → Threshold로 정상 / 이상 판정
#    → 실제 정답과 예측을 비교하여 기록
#
#    이 함수를 정상과 불량 검사에 공통 사용합니다.
# ==================================================
def inspect(
    image_path,
    actual,
    sample_id
):

    # ------------------------------------------
    # ⑤-1. 검사 이미지에서 Feature 추출
    #
    #       새로운 병 이미지 1장
    #                ↓
    #       사전학습된 ResNet18
    #                ↓
    #       Feature Vector 512개
    # ------------------------------------------
    feature = extract_feature(
        model,
        image_path
    )


    # ------------------------------------------
    # ⑤-2. 정상 Reference와 유사도 계산
    #
    #       Reference Feature
    #                 +
    #       새로운 병의 Feature
    #                 ↓
    #       Cosine Similarity
    #
    #       유사도 높음 → 정상 기준과 비슷함
    #       유사도 낮음 → 정상 기준과 차이가 큼
    # ------------------------------------------
    similarity = cosine_similarity(
        reference_feature,
        feature,
    )


    # ------------------------------------------
    # ⑤-3. Threshold로 Normal / Anomaly 판정
    #
    #       Similarity >= Threshold → Normal
    #       Similarity <  Threshold → Anomaly
    #
    #       예: Threshold 약 0.982113
    # ------------------------------------------
    predicted = (
        "Normal"
        if similarity >= threshold
        else "Anomaly"
    )


    # ------------------------------------------
    # ⑤-4. 검사 결과를 리스트에 기록
    #
    #       sample_id : 이미지 구분용 경로
    #       file      : 이미지 파일명
    #       actual    : 실제 정답
    #       similarity: 계산된 유사도
    #       threshold : 판정 기준값
    #       predicted : 검사기의 예측 결과
    #       correct   : 정답 여부
    #
    #       correct = True  → 정확히 판정
    #       correct = False → 잘못 판정
    # ------------------------------------------
    rows.append(
        {
            "sample_id": sample_id,
            "file": image_path.name,
            "actual": actual,
            "similarity": round(
                similarity,
                6
            ),
            "threshold": round(
                threshold,
                6
            ),
            "predicted": predicted,
            "correct": predicted == actual,
        }
    )


# ==================================================
# ⑥ 정상 Common Query 이미지 10장 검사
#
#    → common_query/normal 폴더의 이미지 사용
#    → 실제 정답은 모두 Normal
#    → inspect() 함수를 이용하여 검사
# ==================================================

# ----------------------------------------------
# ⑥-1. 정상 Query 이미지를 한 장씩 가져오기
# ----------------------------------------------
for image_path in sorted(
    (QUERY_ROOT / "normal").glob("*.png")
):

    # ------------------------------------------
    # ⑥-2. 정상 이미지 검사
    #
    #       actual = Normal
    #
    #       sample_id에 normal/을 붙여
    #       불량 이미지와 구분합니다.
    # ------------------------------------------
    inspect(
        image_path,
        "Normal",
        f"normal/{image_path.name}",
    )


# ==================================================
# ⑦ 작은 파손 Common Query 이미지 10장 검사
#
#    → common_query/anomaly/broken_small 사용
#    → 실제 정답은 모두 Anomaly
#    → 정상 검사와 동일한 inspect() 함수 사용
#
#    One-shot은 결함 이미지를 기준으로
#    학습한 것이 아니라 정상 기준과 비교합니다.
# ==================================================

# ----------------------------------------------
# ⑦-1. 작은 파손 Query 폴더 설정
# ----------------------------------------------
anomaly_dir = (
    QUERY_ROOT
    / "anomaly"
    / "broken_small"
)


# ----------------------------------------------
# ⑦-2. 작은 파손 이미지를 한 장씩 검사
# ----------------------------------------------
for image_path in sorted(
    anomaly_dir.glob("*.png")
):

    # ------------------------------------------
    # ⑦-3. 이상 이미지 검사
    #
    #       actual = Anomaly
    #
    #       sample_id에 결함 종류를 기록하여
    #       정상 이미지와 구분합니다.
    # ------------------------------------------
    inspect(
        image_path,
        "Anomaly",
        f"anomaly/broken_small/{image_path.name}",
    )


# ==================================================
# ⑧ 전체 검사 결과를 CSV로 저장
#
#    → 정상과 불량 Query의 판정 결과 기록
#    → 이후 Few-shot 결과와 비교할 때 사용
#
#    저장 파일:
#    reports/day08_one_shot_results.csv
# ==================================================

# ----------------------------------------------
# ⑧-1. CSV 저장 폴더 생성
# ----------------------------------------------
CSV_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ----------------------------------------------
# ⑧-2. CSV 파일 열기
# ----------------------------------------------
with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    # ------------------------------------------
    # ⑧-3. CSV 컬럼명 설정
    #
    #       rows[0].keys()
    #       → 첫 번째 결과의 항목 이름 사용
    # ------------------------------------------
    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys(),
    )

    # ------------------------------------------
    # ⑧-4. CSV 컬럼명 작성
    # ------------------------------------------
    writer.writeheader()

    # ------------------------------------------
    # ⑧-5. 이미지별 판정 결과 저장
    # ------------------------------------------
    writer.writerows(
        rows
    )


# ==================================================
# ⑨ 검사 결과 출력
#
#    → 사용한 Threshold 확인
#    → 이미지별 실제 정답과 예측 결과 출력
#    → Similarity와 정답 여부 확인
#    → 결과 CSV 저장 경로 확인
# ==================================================

# ----------------------------------------------
# ⑨-1. 검사 완료 제목 출력
# ----------------------------------------------
print("=" * 70)
print("DAY08 ONE-SHOT TEST")
print("=" * 70)


# ----------------------------------------------
# ⑨-2. 검사에 사용한 Threshold 출력
# ----------------------------------------------
print(
    "Threshold:",
    round(threshold, 6)
)

print()


# ----------------------------------------------
# ⑨-3. 이미지별 검사 결과 출력
#
#       sample_id  : 이미지 구분
#       actual     : 실제 정답
#       predicted  : 예측 결과
#       similarity : 정상 기준과의 유사도
#       correct    : 정답 여부
# ----------------------------------------------
for row in rows:

    print(
        row["sample_id"],
        "actual=",
        row["actual"],
        "predicted=",
        row["predicted"],
        "similarity=",
        row["similarity"],
        "correct=",
        row["correct"],
    )


# ----------------------------------------------
# ⑨-4. 결과 CSV 저장 경로 출력
# ----------------------------------------------
print()

print(
    "Saved:",
    CSV_PATH
)
