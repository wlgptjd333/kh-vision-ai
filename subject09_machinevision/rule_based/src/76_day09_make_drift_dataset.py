from pathlib import Path
import shutil

import cv2


SOURCE_DIR = Path("samples/day07/test/normal")
BASELINE_DIR = Path("samples/day09/drift/baseline")
LIGHTING_DIR = Path("samples/day09/drift/lighting")
PREVIEW_PATH = Path("outputs/day09/drift/preview.jpg")

LIGHTING_ALPHA = 0.65
MAX_IMAGES = 20


image_files = sorted(SOURCE_DIR.glob("*.png"))[:MAX_IMAGES]

if not image_files:
    raise RuntimeError(
        f"Day07 Normal Test 이미지가 없습니다: {SOURCE_DIR}"
    )

for directory in [BASELINE_DIR, LIGHTING_DIR]:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


for image_path in image_files:
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    baseline_path = BASELINE_DIR / image_path.name
    lighting_path = LIGHTING_DIR / image_path.name

    shutil.copy2(image_path, baseline_path)

    drifted = cv2.convertScaleAbs(
        image,
        alpha=LIGHTING_ALPHA,
        beta=0,
    )

    if not cv2.imwrite(str(lighting_path), drifted):
        raise RuntimeError(
            f"Lighting 이미지를 저장하지 못했습니다: {lighting_path}"
        )


first_name = image_files[0].name
baseline_preview = cv2.imread(str(BASELINE_DIR / first_name))
lighting_preview = cv2.imread(str(LIGHTING_DIR / first_name))

if baseline_preview is None or lighting_preview is None:
    raise RuntimeError("Preview 이미지를 만들 수 없습니다.")

preview = cv2.hconcat(
    [baseline_preview, lighting_preview]
)

PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(PREVIEW_PATH), preview):
    raise RuntimeError("Preview 저장에 실패했습니다.")

print("=" * 70)
print("DAY09 DRIFT DATASET")
print("=" * 70)
print("Source Images :", len(image_files))
print("Baseline      :", len(list(BASELINE_DIR.glob("*.png"))))
print("Lighting      :", len(list(LIGHTING_DIR.glob("*.png"))))
print("Alpha         :", LIGHTING_ALPHA)
print("Preview       :", PREVIEW_PATH)
