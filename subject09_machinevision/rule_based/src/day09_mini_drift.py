from pathlib import Path
import cv2

SOURCE_DIR = Path("samples/day07/test/normal")
BASELINE_DIR = Path("samples/day09/mini_drift/baseline")
BLUR_DIR = Path("samples/day09/mini_drift/blur")

# TODO 1: 원본 정상 이미지 20장을 불러옵니다.
# TODO 2: baseline에는 원본을 복사합니다.
# TODO 3: blur에는 같은 이미지에 Gaussian Blur만 적용합니다.
# TODO 4: Day07 Reference와 Threshold를 그대로 읽습니다.
# TODO 5: 두 조건의 Anomaly Score와 FP Rate를 계산합니다.
# TODO 6: 결과를 CSV 또는 Markdown으로 정리합니다.

print("Day09 Mini Drift를 완성하세요.")
