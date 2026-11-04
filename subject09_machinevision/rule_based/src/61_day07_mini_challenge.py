import json
import shutil
from pathlib import Path

import cv2

from day07_anomaly_core import (
    anomaly_score,
    create_backbone,
    extract_features,
    load_reference,
)


# --------------------------------------------------
# 1. 입력 / 출력 경로
# --------------------------------------------------
NORMAL_DIR = Path(
    "samples/day07/test/normal"
)

OUTPUT_ROOT = Path(
    "samples/day07/mini_challenge"
)

VARIANT_DIR = (
    OUTPUT_ROOT
    / "variants"
)

REFERENCE_PATH = Path(
    "artifacts/day07/normal_reference.npz"
)

RULE_PATH = Path(
    "configs/day07_anomaly_rules.json"
)


# --------------------------------------------------
# 2. 학생이 직접 바꿀 조건
# --------------------------------------------------
DARK_ALPHA = 0.75
BLUR_KERNEL = 7
ROTATION_ANGLE = 10.0


# --------------------------------------------------
# 3. 이전 결과 초기화
# --------------------------------------------------
if OUTPUT_ROOT.exists():
    shutil.rmtree(
        OUTPUT_ROOT
    )

VARIANT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 4. 앞 Challenge와 다른 정상 이미지 선택
# --------------------------------------------------
normal_files = sorted(
    NORMAL_DIR.glob("*.png")
)

if len(normal_files) < 2:
    raise RuntimeError(
        "Mini Challenge용 Normal 이미지가 부족합니다."
    )

source_path = normal_files[1]

image = cv2.imread(
    str(source_path)
)

if image is None:
    raise FileNotFoundError(
        source_path
    )

height, width = image.shape[:2]


# --------------------------------------------------
# 5. Dark
# --------------------------------------------------
variant = cv2.convertScaleAbs(
    image,
    alpha=DARK_ALPHA,
    beta=0,
)

cv2.imwrite(
    str(
        VARIANT_DIR
        / "mini_dark.png"
    ),
    variant,
)


# --------------------------------------------------
# 6. Blur
# --------------------------------------------------
variant = cv2.GaussianBlur(
    image,
    (
        BLUR_KERNEL,
        BLUR_KERNEL,
    ),
    0,
)

cv2.imwrite(
    str(
        VARIANT_DIR
        / "mini_blur.png"
    ),
    variant,
)


# --------------------------------------------------
# 7. Rotation
# --------------------------------------------------
center = (
    width // 2,
    height // 2,
)

matrix = cv2.getRotationMatrix2D(
    center,
    ROTATION_ANGLE,
    1.0,
)

variant = cv2.warpAffine(
    image,
    matrix,
    (width, height),
    borderMode=cv2.BORDER_REFLECT,
)

cv2.imwrite(
    str(
        VARIANT_DIR
        / "mini_rotation.png"
    ),
    variant,
)


# --------------------------------------------------
# 8. Frozen Rule 불러오기
# --------------------------------------------------
rules = json.loads(
    RULE_PATH.read_text(
        encoding="utf-8"
    )
)

threshold = float(
    rules["anomaly_threshold"]
)

reference_vector, _ = load_reference(
    REFERENCE_PATH
)

model = create_backbone()


# --------------------------------------------------
# 9. 세 이미지 검사
# --------------------------------------------------
print("=" * 70)
print("DAY07 MINI CHALLENGE")
print("=" * 70)

print(
    "Source   :",
    source_path,
)

print(
    "Threshold:",
    threshold,
)

print()


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

    result = (
        "FP"
        if predicted == "Anomaly"
        else "PASS"
    )

    print(
        image_path.name,
        "score=",
        round(score, 6),
        "predicted=",
        predicted,
        "result=",
        result,
    )
