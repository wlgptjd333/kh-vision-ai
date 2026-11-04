# ==================================================
# ① 필요한 라이브러리와 공통 함수 불러오기
#
#    → csv   : 검사 결과를 CSV로 저장
#    → Path  : 파일과 폴더 경로 관리
#    → NumPy : 저장된 Prototype 불러오기
#
#    day08_feature_core의 공통 함수 사용
#    → create_feature_model() : ResNet18 준비
#    → extract_feature()      : 이미지 특징 추출
#    → cosine_similarity()    : 유사도 계산
# ==================================================
import csv
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
#    → 66번에서 저장한 두 Prototype 사용
#    → 검사 결과를 CSV 파일로 저장
# ==================================================

# ----------------------------------------------
# ②-1. 공통 검사 이미지 폴더
#
#       normal/                 : 정상 10장
#       anomaly/broken_small/   : 작은 파손 10장
# ----------------------------------------------
QUERY_ROOT = Path(
    "samples/day08/common_query"
)


# ----------------------------------------------
# ②-2. 66번에서 저장한 Prototype 파일
#
#       Normal Prototype 1개
#       Anomaly Prototype 1개
# ----------------------------------------------
ARTIFACT_PATH = Path(
    "artifacts/day08/fewshot_prototypes.npz"
)


# ----------------------------------------------
# ②-3. Few-shot 검사 결과 CSV 저장 경로
# ----------------------------------------------
CSV_PATH = Path(
    "reports/day08_fewshot_results.csv"
)


# ==================================================
# ③ 66번에서 만든 두 Prototype 불러오기
#
#    → NPZ 파일에는 두 개의 Feature Vector 저장
#
#    normal_prototype
#    → 정상 Support 5장의 대표 특징
#
#    anomaly_prototype
#    → 작은 파손 Support 5장의 대표 특징
#
#    각각 512차원 Feature Vector입니다.
# ==================================================

# ----------------------------------------------
# ③-1. NPZ 파일 불러오기
# ----------------------------------------------
prototype_data = np.load(
    ARTIFACT_PATH
)


# ----------------------------------------------
# ③-2. Normal Prototype 가져오기
#       → 정상 이미지의 대표 특징
# ----------------------------------------------
normal_prototype = prototype_data[
    "normal_prototype"
]


# ----------------------------------------------
# ③-3. Anomaly Prototype 가져오기
#       → 작은 파손 이미지의 대표 특징
# ----------------------------------------------
anomaly_prototype = prototype_data[
    "anomaly_prototype"
]


# ==================================================
# ④ Feature 추출용 ResNet18 준비
#
#    → 사전학습된 ResNet18 사용
#    → 새로 모델을 학습하지 않음
#    → Query 이미지의 512차원 특징 추출
#
#    rows에는 이미지별 검사 결과를 기록합니다.
# ==================================================

# ----------------------------------------------
# ④-1. 특징 추출용 ResNet18 준비
# ----------------------------------------------
model = create_feature_model()


# ----------------------------------------------
# ④-2. 검사 결과를 저장할 빈 리스트 생성
# ----------------------------------------------
rows = []


# ==================================================
# ⑤ 이미지 한 장을 검사하는 함수 작성
#
#    함수: inspect()
#
#    처리 과정:
#    1) Query 이미지의 Feature 추출
#    2) Normal Prototype과 유사도 계산
#    3) Anomaly Prototype과 유사도 계산
#    4) 유사도가 더 높은 쪽으로 판정
#    5) 실제 정답과 비교하여 결과 기록
#
#    One-shot과 다른 점!
#    → Threshold와 비교하지 않습니다.
#    → 두 Prototype의 유사도를 직접 비교합니다.
# ==================================================
def inspect(
    image_path,
    actual,
    sample_id
):

    # ----------------------------------------------
    # ⑤-1. Query 이미지의 Feature 추출
    #
    #       새로운 병 이미지 1장
    #                 ↓
    #       사전학습된 ResNet18
    #                 ↓
    #       512차원 Feature Vector
    # ----------------------------------------------
    feature = extract_feature(
        model,
        image_path
    )


    # ----------------------------------------------
    # ⑤-2. Normal Prototype과 유사도 계산
    #
    #       Query Feature
    #             ↕
    #       Normal Prototype
    #             ↓
    #       normal_similarity
    # ----------------------------------------------
    normal_similarity = cosine_similarity(
        feature,
        normal_prototype,
    )


    # ----------------------------------------------
    # ⑤-3. Anomaly Prototype과 유사도 계산
    #
    #       Query Feature
    #             ↕
    #       Anomaly Prototype
    #             ↓
    #       anomaly_similarity
    # ----------------------------------------------
    anomaly_similarity = cosine_similarity(
        feature,
        anomaly_prototype,
    )


    # ----------------------------------------------
    # ⑤-4. 두 유사도를 비교하여 판정
    #
    #       Normal 유사도 >= Anomaly 유사도
    #       → Normal
    #
    #       Normal 유사도 < Anomaly 유사도
    #       → Anomaly
    #
    #       유사도가 같은 경우에는
    #       현재 코드에서 Normal로 판정합니다.
    #
    #       별도의 Threshold는 사용하지 않습니다.
    # ----------------------------------------------
    predicted = (
        "Normal"
        if normal_similarity >= anomaly_similarity
        else "Anomaly"
    )


    # ----------------------------------------------
    # ⑤-5. 검사 결과를 리스트에 기록
    #
    #       sample_id          : 이미지 식별자
    #       file               : 이미지 파일명
    #       actual             : 실제 정답
    #       normal_similarity  : 정상 대표와 유사도
    #       anomaly_similarity : 불량 대표와 유사도
    #       predicted          : 예측 결과
    #       correct            : 정답 여부
    #
    #       True  → 정확하게 판정
    #       False → 잘못 판정
    # ----------------------------------------------
    rows.append(
        {
            "sample_id": sample_id,

            "file": image_path.name,

            "actual": actual,

            "normal_similarity": round(
                normal_similarity,
                6
            ),

            "anomaly_similarity": round(
                anomaly_similarity,
                6
            ),

            "predicted": predicted,

            "correct": predicted == actual,
        }
    )


# ==================================================
# ⑥ 정상 Common Query 이미지 검사
#
#    → 정상 Query 이미지 10장 사용
#    → 실제 정답은 모두 Normal
#    → inspect() 함수로 한 장씩 검사
# ==================================================

# ----------------------------------------------
# ⑥-1. 정상 Query 이미지 목록 반복
# ----------------------------------------------
for image_path in sorted(
    (QUERY_ROOT / "normal").glob("*.png")
):

    # ------------------------------------------
    # ⑥-2. 정상 이미지 검사
    #
    #       actual = Normal
    #
    #       sample_id에 normal/을 기록하여
    #       검사 이미지의 종류를 구분합니다.
    # ------------------------------------------
    inspect(
        image_path,
        "Normal",
        f"normal/{image_path.name}",
    )


# ==================================================
# ⑦ 작은 파손 Common Query 이미지 검사
#
#    → 작은 파손 Query 이미지 10장 사용
#    → 실제 정답은 모두 Anomaly
#
#    65번 One-shot에서 사용한 것과
#    동일한 Common Query 이미지입니다.
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
# ⑦-2. 작은 파손 이미지 반복
# ----------------------------------------------
for image_path in sorted(
    anomaly_dir.glob("*.png")
):

    # ------------------------------------------
    # ⑦-3. 작은 파손 이미지 검사
    #
    #       actual = Anomaly
    #
    #       sample_id에 anomaly/broken_small/
    #       경로를 기록합니다.
    # ------------------------------------------
    inspect(
        image_path,
        "Anomaly",
        f"anomaly/broken_small/{image_path.name}",
    )


# ==================================================
# ⑧ 검사 결과 CSV 파일 저장
#
#    → 정상 10장과 작은 파손 10장 결과 기록
#    → 이미지별 두 Prototype 유사도 기록
#    → 실제 정답과 예측 결과 기록
#
#    저장 결과는 다음 68번 비교에서 사용합니다.
# ==================================================

# ----------------------------------------------
# ⑧-1. 저장할 reports 폴더 생성
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
    # ⑧-3. 결과 CSV 컬럼명 설정
    #
    #       rows[0].keys()
    #       → 첫 번째 검사 결과의 항목 이름 사용
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
    # ⑧-5. 전체 검사 결과 저장
    # ------------------------------------------
    writer.writerows(
        rows
    )


# ==================================================
# ⑨ 실행 결과 출력
#
#    → 이미지별 실제 정답 출력
#    → Normal / Anomaly 예측 출력
#    → 정상·불량 Prototype 유사도 출력
#    → 정답 여부 출력
#    → 결과 CSV 저장 경로 출력
#
#    전체 Accuracy 비교는 68번에서 진행합니다.
# ==================================================

# ----------------------------------------------
# ⑨-1. 검사 완료 제목 출력
# ----------------------------------------------
print("=" * 70)
print("DAY08 FEW-SHOT TEST")
print("=" * 70)


# ----------------------------------------------
# ⑨-2. 이미지별 검사 결과 출력
# ----------------------------------------------
for row in rows:

    print(
        row["sample_id"],

        "actual=",
        row["actual"],

        "predicted=",
        row["predicted"],

        "normal_sim=",
        row["normal_similarity"],

        "anomaly_sim=",
        row["anomaly_similarity"],

        "correct=",
        row["correct"],
    )


# ----------------------------------------------
# ⑨-3. CSV 저장 경로 출력
# ----------------------------------------------
print()

print(
    "Saved:",
    CSV_PATH
)
