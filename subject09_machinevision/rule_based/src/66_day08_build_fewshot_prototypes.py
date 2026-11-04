# ==================================================
# ① 필요한 라이브러리와 공통 함수 불러오기
#
#    → json  : Few-shot 검사 설정을 JSON으로 저장
#    → Path  : 파일과 폴더 경로 관리
#    → NumPy : Normal / Anomaly Prototype 저장
#
#    day08_feature_core의 공통 함수 사용
#    → create_feature_model() : ResNet18 준비
#    → extract_feature()      : 이미지 특징 추출
#    → build_prototype()      : 여러 특징의 대표값 생성
# ==================================================
import json
from pathlib import Path

import numpy as np

from day08_feature_core import (
    build_prototype,
    create_feature_model,
    extract_feature,
)


# ==================================================
# ② 입력 / 출력 경로 설정
#
#    → 입력 : 63번에서 분리한 Few-shot Support
#    → 출력 : 정상·불량 Prototype 저장 파일
#    → 출력 : Few-shot 검사 설정 JSON 파일
# ==================================================

# ----------------------------------------------
# ②-1. Few-shot Support 이미지 폴더
#
#       normal/  : 정상 이미지 5장
#       anomaly/ : 작은 파손 이미지 5장
# ----------------------------------------------
SUPPORT_ROOT = Path(
    "samples/day08/few_shot/support"
)

# ----------------------------------------------
# ②-2. Normal / Anomaly Prototype 저장 경로
#
#       두 그룹의 대표 Feature를
#       하나의 NPZ 파일에 저장합니다.
# ----------------------------------------------
ARTIFACT_PATH = Path(
    "artifacts/day08/fewshot_prototypes.npz"
)

# ----------------------------------------------
# ②-3. Few-shot 검사 설정 JSON 저장 경로
# ----------------------------------------------
CONFIG_PATH = Path(
    "configs/day08_fewshot_config.json"
)


# ==================================================
# ③ 특징 추출용 ResNet18 준비
#
#    → 공통 함수 create_feature_model() 사용
#    → ImageNet 사전학습 ResNet18 불러오기
#    → 이미지마다 512차원 Feature Vector 추출
#
#    이번에도 ResNet18을 새로 학습하지 않습니다.
# ==================================================
model = create_feature_model()


# ==================================================
# ④ Support 폴더의 Feature를 추출하는 함수
#
#    함수: extract_folder()
#
#    → 폴더에서 PNG 이미지 목록 가져오기
#    → 이미지가 없으면 오류 발생
#    → 각 이미지에서 Feature Vector 추출
#    → 이미지 목록과 Feature 목록 반환
#
#    이 함수를 정상·불량 그룹에 공통 사용합니다.
# ==================================================
def extract_folder(folder):

    # ----------------------------------------------
    # ④-1. 해당 폴더의 PNG 이미지 목록 읽기
    # ----------------------------------------------
    files = sorted(
        folder.glob("*.png")
    )

    # ----------------------------------------------
    # ④-2. Support 이미지 존재 여부 확인
    #
    #       이미지가 없다면 Prototype을
    #       만들 수 없으므로 오류 발생
    # ----------------------------------------------
    if not files:
        raise RuntimeError(
            f"Support 이미지가 없습니다: {folder}"
        )

    # ----------------------------------------------
    # ④-3. 이미지마다 Feature Vector 추출
    #
    #       예: 정상 이미지 5장
    #           ↓
    #       ResNet18 특징 추출
    #           ↓
    #       512개 특징값을 가진 Vector 5개
    #
    #       공통 함수 extract_feature() 사용
    # ----------------------------------------------
    features = [
        extract_feature(
            model,
            image_path
        )
        for image_path in files
    ]

    # ----------------------------------------------
    # ④-4. 이미지 목록과 Feature 목록 반환
    # ----------------------------------------------
    return files, features


# ==================================================
# ⑤ 정상 / 불량 Support 이미지에서 Feature 추출
#
#    → 정상 이미지 5장의 Feature 추출
#    → 작은 파손 이미지 5장의 Feature 추출
#
#    이후 두 그룹의 Prototype을 따로 만듭니다.
# ==================================================

# ----------------------------------------------
# ⑤-1. 정상 Support 이미지의 특징 추출
#
#       samples/day08/few_shot/support/normal
#       → 정상 이미지 5장
#       → 정상 Feature Vector 5개
# ----------------------------------------------
normal_files, normal_features = extract_folder(
    SUPPORT_ROOT / "normal"
)

# ----------------------------------------------
# ⑤-2. 불량 Support 이미지의 특징 추출
#
#       samples/day08/few_shot/support/anomaly
#       → 작은 파손 이미지 5장
#       → 불량 Feature Vector 5개
# ----------------------------------------------
anomaly_files, anomaly_features = extract_folder(
    SUPPORT_ROOT / "anomaly"
)


# ==================================================
# ⑥ Normal / Anomaly Prototype 생성
#
#    → 공통 함수 build_prototype() 사용
#    → 같은 그룹의 Feature Vector를 평균
#    → 평균 Feature를 L2 정규화
#
#    Normal Prototype  : 정상 병의 대표 특징
#    Anomaly Prototype : 작은 파손 병의 대표 특징
#
#    각 Prototype의 최종 형태는 (512,)입니다.
# ==================================================

# ----------------------------------------------
# ⑥-1. 정상 Feature 5개의 대표 특징 생성
# ----------------------------------------------
normal_prototype = build_prototype(
    normal_features
)

# ----------------------------------------------
# ⑥-2. 작은 파손 Feature 5개의 대표 특징 생성
# ----------------------------------------------
anomaly_prototype = build_prototype(
    anomaly_features
)


# ==================================================
# ⑦ 생성한 두 Prototype을 NPZ 파일로 저장
#
#    → Normal Prototype 저장
#    → Anomaly Prototype 저장
#
#    NPZ는 여러 NumPy 배열을
#    하나의 파일에 저장할 수 있는 형식입니다.
#
#    이후 67번 코드에서 불러와 사용합니다.
# ==================================================

# ----------------------------------------------
# ⑦-1. 저장 폴더가 없으면 생성
# ----------------------------------------------
ARTIFACT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------
# ⑦-2. 두 Prototype을 하나의 NPZ로 저장
#
#       저장되는 배열 이름:
#       normal_prototype
#       anomaly_prototype
# ----------------------------------------------
np.savez_compressed(
    ARTIFACT_PATH,
    normal_prototype=normal_prototype,
    anomaly_prototype=anomaly_prototype,
)


# ==================================================
# ⑧ Few-shot 검사 설정을 JSON으로 저장
#
#    → Baseline 버전 기록
#    → 사용한 데이터셋과 모델 기록
#    → 정상 / 불량 Support 이미지 수 기록
#    → 불량 예제의 종류 기록
#    → 검사 판정 규칙 기록
#
#    Few-shot의 판정 방법:
#    → Query와 두 Prototype의 유사도 비교
#    → 유사도가 더 높은 그룹으로 판정
#
#    One-shot과 달리 이번 코드에서는
#    별도의 단일 Threshold를 계산하지 않습니다.
# ==================================================

# ----------------------------------------------
# ⑧-1. Few-shot 검사 설정 정보 구성
# ----------------------------------------------
config = {

    # Few-shot Baseline 버전
    "baseline_version": "day08-few-shot-v1.0",

    # 사용 데이터셋
    "dataset": "MVTec AD bottle",

    # 검사 방법
    "method": "prototype_cosine_similarity",

    # 이미지 특징 추출 모델
    "backbone": "resnet18_imagenet_pretrained",

    # 정상 Support 이미지 수
    "normal_support_count": len(normal_files),

    # 불량 Support 이미지 수
    "anomaly_support_count": len(anomaly_files),

    # 사용한 불량 예제 종류
    "anomaly_support_type": "broken_small",

    # Query 검사 시 사용할 판정 규칙
    "decision_rule": "choose prototype with higher similarity",
}

# ----------------------------------------------
# ⑧-2. configs 폴더가 없으면 생성
# ----------------------------------------------
CONFIG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------
# ⑧-3. 검사 설정을 JSON 파일로 저장
#
#       이후 67번에서 확인할
#       Few-shot 검사 조건을 기록합니다.
# ----------------------------------------------
CONFIG_PATH.write_text(
    json.dumps(
        config,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ==================================================
# ⑨ 실행 결과 확인
#
#    → 사용한 정상 Support 이미지 수 출력
#    → 사용한 불량 Support 이미지 수 출력
#    → 불량 종류 출력
#    → NPZ 저장 경로 출력
#    → JSON 저장 경로 출력
#
#    아직 새로운 Query에 대한
#    Normal / Anomaly 판정은 수행하지 않습니다.
# ==================================================

# ----------------------------------------------
# ⑨-1. Few-shot Prototype 생성 완료 제목
# ----------------------------------------------
print("=" * 70)
print("DAY08 FEW-SHOT PROTOTYPE")
print("=" * 70)

# ----------------------------------------------
# ⑨-2. 정상 Support 이미지 수 출력
# ----------------------------------------------
print(
    "Normal Support  :",
    len(normal_files)
)

# ----------------------------------------------
# ⑨-3. 불량 Support 이미지 수 출력
# ----------------------------------------------
print(
    "Anomaly Support :",
    len(anomaly_files)
)

# ----------------------------------------------
# ⑨-4. 불량 Support의 결함 종류 출력
# ----------------------------------------------
print(
    "Anomaly Type    : broken_small"
)

# ----------------------------------------------
# ⑨-5. Prototype NPZ 저장 경로 출력
# ----------------------------------------------
print(
    "Saved Artifact  :",
    ARTIFACT_PATH
)

# ----------------------------------------------
# ⑨-6. 설정 JSON 저장 경로 출력
# ----------------------------------------------
print(
    "Saved Config    :",
    CONFIG_PATH
)
