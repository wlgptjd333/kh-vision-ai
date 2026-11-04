import shutil
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. 입력 / 출력 경로
# --------------------------------------------------
NORMAL_DIR = Path(
    "samples/day07/test/normal"
)

CHALLENGE_ROOT = Path(
    "samples/day07/challenge"
)

SOURCE_DIR = (
    CHALLENGE_ROOT
    / "source"
)

VARIANT_DIR = (
    CHALLENGE_ROOT
    / "variants"
)


# --------------------------------------------------
# 2. 이전 Challenge 결과 초기화
# --------------------------------------------------
if CHALLENGE_ROOT.exists():
    shutil.rmtree(
        CHALLENGE_ROOT
    )

SOURCE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VARIANT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 3. Normal Test 이미지 선택
# --------------------------------------------------
normal_files = sorted(
    NORMAL_DIR.glob("*.png")
)

if not normal_files:
    raise RuntimeError(
        "Test Normal 이미지가 없습니다."
    )


# 첫 번째 Normal Test 이미지를 사용합니다.
source_path = normal_files[0]

source_copy = (
    SOURCE_DIR
    / source_path.name
)

shutil.copy2(
    source_path,
    source_copy,
)


image = cv2.imread(
    str(source_path)
)

if image is None:
    raise FileNotFoundError(
        source_path
    )


height, width = image.shape[:2]


# --------------------------------------------------
# 4. Dark
# --------------------------------------------------
variant = cv2.convertScaleAbs(
    image,
    alpha=0.60,
    beta=0,
)

cv2.imwrite(
    str(
        VARIANT_DIR
        / "normal_dark.png"
    ),
    variant,
)


# --------------------------------------------------
# 5. Bright
# --------------------------------------------------
variant = cv2.convertScaleAbs(
    image,
    alpha=1.20,
    beta=20,
)

cv2.imwrite(
    str(
        VARIANT_DIR
        / "normal_bright.png"
    ),
    variant,
)


# --------------------------------------------------
# 6. Blur
# --------------------------------------------------
variant = cv2.GaussianBlur(
    image,
    (11, 11),
    0,
)

cv2.imwrite(
    str(
        VARIANT_DIR
        / "normal_blur.png"
    ),
    variant,
)


# --------------------------------------------------
# 7. Position Shift
# --------------------------------------------------
matrix = np.float32(
    [
        [1, 0, 35],
        [0, 1, 20],
    ]
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
        / "normal_shift.png"
    ),
    variant,
)


# --------------------------------------------------
# 8. Rotation
# --------------------------------------------------
center = (
    width // 2,
    height // 2,
)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    6.0,
    1.0,
)

variant = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height),
    borderMode=cv2.BORDER_REFLECT,
)

cv2.imwrite(
    str(
        VARIANT_DIR
        / "normal_rotation.png"
    ),
    variant,
)


# --------------------------------------------------
# 9. 결과 확인
# --------------------------------------------------
print(
    "Challenge Source:",
    source_path,
)

print(
    "Saved:",
    VARIANT_DIR,
)
