import csv

import cv2

from day06_ocr_utils import (
    PROJECT_ROOT,
    load_rules,
    preprocess_for_ocr,
    run_ocr,
)

from day06_ocr_validation import (
    clean_ocr_text,
    validate_pattern,
)


# ========================================
# 경로 설정
# ========================================

PRODUCT_DIR = (
    PROJECT_ROOT
    / "samples"
    / "day06"
    / "mini_challenge"
    / "product"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "day06"
    / "mini_challenge"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_mini_challenge_results.csv"
)


# ========================================
# 기존 OCR 설정 가져오기
# ========================================

rules = load_rules()


# ========================================
# Ch.-B. 검사에 맞게 변경
# ========================================

rules[
    "preprocess_mode"
] = "original"

rules[
    "ocr_psm"
] = 7

rules[
    "whitelist"
] = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789-"
)

rules[
    "expected_pattern"
] = r"^[A-Z0-9-]+$"


# ========================================
# Mini Challenge 이미지 찾기
# ========================================

image_paths = sorted(
    [
        path
        for path in PRODUCT_DIR.iterdir()
        if path.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
        }
    ]
)


if not image_paths:
    raise FileNotFoundError(
        f"Mini Challenge 이미지가 없습니다: "
        f"{PRODUCT_DIR}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


rows = []


# ========================================
# 4장 하나씩 검사
# ========================================

for image_path in image_paths:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        continue


    print()
    print(
        "=" * 60
    )

    print(
        "Image:",
        image_path.name
    )

    print(
        "Ch.-B.의 코드 값만 "
        "마우스로 선택하세요."
    )


    # ====================================
    # Ch.-B. ROI 직접 선택
    # ====================================

    x, y, w, h = cv2.selectROI(
        f"Select Ch.-B. ROI - {image_path.name}",
        image,
        fromCenter=False,
        showCrosshair=True,
    )

    cv2.destroyAllWindows()


    if w == 0 or h == 0:
        print(
            "ROI 선택이 취소되었습니다."
        )
        continue


    roi = image[
        int(y):int(y + h),
        int(x):int(x + w),
    ]


    # ====================================
    # 앞에서 선택한 original 사용
    # ====================================

    processed = (
        preprocess_for_ocr(
            roi,
            rules,
        )
    )


    # ====================================
    # 이전 실습과 동일하게 4배 확대
    # ====================================

    ocr_input = cv2.resize(
        processed,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_CUBIC,
    )


    # ====================================
    # OCR
    # ====================================

    raw_text = run_ocr(
        ocr_input,
        rules,
    )

    ocr_text = (
        clean_ocr_text(
            raw_text
        )
        .upper()
    )


    # ====================================
    # 이미지에서 직접 확인한 정답 입력
    # ====================================

    print(
        "OCR 결과:",
        ocr_text
    )

    expected_text = input(
        "이미지에 실제로 보이는 "
        "Ch.-B. 값을 입력하세요: "
    ).strip().upper()


    # ====================================
    # 형식과 OCR 정확도 확인
    # ====================================

    format_ok = (
        validate_pattern(
            ocr_text,
            rules[
                "expected_pattern"
            ],
        )
    )

    exact_match = (
        ocr_text
        == expected_text
    )


    # ====================================
    # OCR 입력 이미지 저장
    # ====================================

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_chb_ocr_input.jpg"
    )

    cv2.imwrite(
        str(output_path),
        ocr_input,
    )


    # ====================================
    # 결과 기록
    # ====================================

    rows.append(
        {
            "file":
                image_path.name,

            "expected_text":
                expected_text,

            "ocr_text":
                ocr_text,

            "format_ok":
                format_ok,

            "exact_match":
                exact_match,
        }
    )


    print(
        "Format OK:",
        format_ok
    )

    print(
        "Exact Match:",
        exact_match
    )


# ========================================
# CSV 저장
# ========================================

if not rows:
    raise RuntimeError(
        "검사 결과가 없습니다."
    )


REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with REPORT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "file",
            "expected_text",
            "ocr_text",
            "format_ok",
            "exact_match",
        ],
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


print()
print(
    "Saved:",
    REPORT_PATH
)

print(
    "Day06 Mini Challenge Complete"
)
