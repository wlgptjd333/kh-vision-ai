from pathlib import Path

import cv2


# [Mini Challenge] 04_batch_roi.py 복사본 — 입력·출력 경로와 좌표만 변경
input_dir = Path("samples/challenge_crown_cap")
output_dir = Path("outputs/challenge_roi")
output_dir.mkdir(parents=True, exist_ok=True)

# 자신의 이미지에 맞게 수정합니다.
# v1: (500, 110, 740, 230) — frame_00090에 맞춤 → frame_00060에서 뚜껑 왼쪽이 잘림
# v2: 왼쪽 70px · 위아래 20~30px 여유 → frame_00060~00240 모두 뚜껑/병 입구가 들어옴
x1 = 430
y1 = 90
x2 = 760
y2 = 260

extensions = ("*.jpg", "*.jpeg", "*.png")

image_files = []
for pattern in extensions:
    image_files.extend(input_dir.glob(pattern))

image_files = sorted(image_files)

print("전체 이미지 수:", len(image_files))

if not image_files:
    raise RuntimeError(
        "samples 폴더에 jpg/jpeg/png 이미지가 없습니다."
    )

success_count = 0

for image_path in image_files:
    image = cv2.imread(str(image_path))

    if image is None:
        print("읽기 실패:", image_path.name)
        continue

    height, width = image.shape[:2]

    cx1 = max(0, min(x1, width))
    cx2 = max(0, min(x2, width))
    cy1 = max(0, min(y1, height))
    cy2 = max(0, min(y2, height))

    if cx1 >= cx2 or cy1 >= cy2:
        print("ROI 실패:", image_path.name)
        continue

    roi = image[cy1:cy2, cx1:cx2]

    roi_path = output_dir / f"{image_path.stem}_roi.jpg"
    preview_path = output_dir / f"{image_path.stem}_preview.jpg"

    cv2.imwrite(str(roi_path), roi)

    preview = image.copy()
    cv2.rectangle(
        preview,
        (cx1, cy1),
        (cx2, cy2),
        (0, 255, 0),
        3,
    )
    cv2.imwrite(str(preview_path), preview)

    success_count += 1
    print(image_path.name, "→", roi_path.name)

print("처리 완료:", success_count, "/", len(image_files))
