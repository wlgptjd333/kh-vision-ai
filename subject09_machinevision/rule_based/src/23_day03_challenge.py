"""3일차 Mini Challenge — 유리병 왕관 뚜껑(빨간색) 있음 / 없음 검사

검사 대상: 1일차 sample.mp4의 갈색 유리병 입구 ROI (06_video_roi.py 결과)
  ok_01 = frame_00090 (뚜껑 있음)   ng_01 = frame_00210 (뚜껑 없음)  → Rule을 만든 이미지
  ok_02 = frame_00060 (뚜껑 있음)   ng_02 = frame_00240 (뚜껑 없음)  → Rule을 고정한 뒤 확인한 이미지

1일차 가설: "병 입구 ROI에서 빨간색 비율을 재면 뚜껑 유무를 구분할 수 있다"를 확인한다.
"""
from pathlib import Path

import cv2
import numpy as np


image_dir = Path(
    "samples/day03/challenge"
)

output_dir = Path(
    "outputs/day03/challenge"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

# Rule을 만든 이미지 → Rule을 고정한 뒤 확인한 이미지 순서
images = [
    ("ok_01.jpg", "OK"),
    ("ng_01.jpg", "NG"),
    ("ok_02.jpg", "OK"),
    ("ng_02.jpg", "NG"),
]


# ------------------------------------------
# TODO 1
# 검사 문제에 맞는 전처리를 선택합니다.
#
# 빨간 뚜껑 vs 투명 유리 → 밝기(Gray)보다 색(HSV)이 차이를 만든다.
# 빨간색은 Hue가 0 근처와 180 근처 두 곳에 있으므로 Mask를 두 번 만들어 합친다.
# S·V 하한으로 흰 배경·회색 유리의 반사를 뺀다.
# ------------------------------------------

def red_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    high_red = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    return cv2.bitwise_or(low_red, high_red)


# ------------------------------------------
# TODO 2
# 필요한 경우 Morphology를 적용합니다.
#
# 유리 반사·갈색 음료 가장자리의 작은 붉은 점을 Opening(3×3)으로 지운다.
# ------------------------------------------

KERNEL = np.ones((3, 3), np.uint8)


def clean(mask):
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)


# ------------------------------------------
# TODO 3
# Contour를 찾고 필요한 Feature를 계산합니다.
#
# largest_area_ratio = 가장 큰 빨간 덩어리 면적 / ROI 면적
# (ROI 크기가 달라도 비교할 수 있도록 비율로)
# ------------------------------------------

MIN_CONTOUR_AREA_RATIO = 0.001


def measure(mask):
    height, width = mask.shape
    roi_area = height * width
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA_RATIO * roi_area]
    if not valid:
        return 0.0, 0, None
    largest = max(valid, key=cv2.contourArea)
    return cv2.contourArea(largest) / roi_area, len(valid), largest


# ------------------------------------------
# TODO 4
# 실제 측정값을 확인한 뒤 Rule을 작성합니다.
#
# 측정 (ok_01 / ng_01):
#   ok_01 largest_area_ratio = 0.0935
#   ng_01 largest_area_ratio = 0.0030
# → 두 값의 중간 0.048을 기준으로 고정한다. (ok_02·ng_02를 본 뒤 바꾸지 않는다)
# ------------------------------------------

RED_CAP_AREA_RATIO_THRESHOLD = 0.048


def judge(largest_area_ratio):
    if largest_area_ratio >= RED_CAP_AREA_RATIO_THRESHOLD:
        return "OK", ""
    return "NG", "NG_RED_CAP_AREA"


correct = 0

for name, expected in images:
    image_path = image_dir / name
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    mask = clean(red_mask(image))
    ratio, count, largest = measure(mask)
    decision, reason = judge(ratio)
    correct += decision == expected

    result = image.copy()
    if largest is not None:
        cv2.drawContours(result, [largest], -1, (0, 255, 0), 2)
    color = (0, 180, 0) if decision == "OK" else (0, 0, 255)
    cv2.putText(result, f"{decision} {ratio:.3f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(str(output_dir / f"{image_path.stem}_mask.jpg"), mask)
    cv2.imwrite(str(output_dir / f"{image_path.stem}_result.jpg"), result)

    print(
        f"{name:10s} expected={expected}  decision={decision:2s}  "
        f"largest_area_ratio={ratio:.4f}  valid_count={count}  reason={reason}"
    )

print("Threshold:", RED_CAP_AREA_RATIO_THRESHOLD)
print("올바른 판정:", correct, "/", len(images))
