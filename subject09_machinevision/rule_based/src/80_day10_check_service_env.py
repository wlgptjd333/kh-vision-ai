from pathlib import Path
import json
import shutil

import cv2
import fastapi
import httpx
import pytesseract
import uvicorn


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PRODUCT_DIR = (
    PROJECT_ROOT
    / "samples"
    / "day06"
    / "product"
)

OCR_RULE_PATH = (
    PROJECT_ROOT
    / "configs"
    / "day06_ocr_rules.json"
)

LABEL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "day06_labels.csv"
)

SERVICE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "day10"
    / "service_config.json"
)

REQUIRED_FILES = [
    OCR_RULE_PATH,
    LABEL_PATH,
    SERVICE_CONFIG_PATH,
]


print("=" * 70)
print("DAY10 SERVICE ENV CHECK")
print("=" * 70)

print("OpenCV   :", cv2.__version__)
print("FastAPI  :", fastapi.__version__)
print("Uvicorn  :", uvicorn.__version__)
print("HTTPX    :", httpx.__version__)
print("Tesseract:", shutil.which("tesseract"))

if shutil.which("tesseract") is None:
    raise RuntimeError(
        "Tesseract Engine이 없습니다."
    )


for path in REQUIRED_FILES:
    if not path.exists():
        raise FileNotFoundError(path)

    print("[OK]", path)


if not PRODUCT_DIR.exists():
    raise FileNotFoundError(PRODUCT_DIR)


image_paths = sorted(
    [
        path
        for path in PRODUCT_DIR.iterdir()
        if path.suffix.lower()
        in {".png", ".jpg", ".jpeg"}
    ]
)

if not image_paths:
    raise RuntimeError(
        "Day06 제품 이미지가 없습니다."
    )


ocr_rules = json.loads(
    OCR_RULE_PATH.read_text(
        encoding="utf-8"
    )
)

service_config = json.loads(
    SERVICE_CONFIG_PATH.read_text(
        encoding="utf-8"
    )
)


required_rule_keys = {
    "baseline_version",
    "roi_by_file",
    "preprocess_mode",
    "ocr_lang",
    "ocr_psm",
    "whitelist",
    "expected_pattern",
}

missing = (
    required_rule_keys
    - set(ocr_rules.keys())
)

if missing:
    raise RuntimeError(
        "Day06 OCR Rule Key가 부족합니다: "
        f"{sorted(missing)}"
    )


print()
print("Product Images :", len(image_paths))
print(
    "OCR Version    :",
    ocr_rules["baseline_version"],
)
print(
    "Service Version:",
    service_config["service_version"],
)
print(
    "Tesseract Ver  :",
    pytesseract.get_tesseract_version(),
)

print()
print("Day10 Service Environment OK")
