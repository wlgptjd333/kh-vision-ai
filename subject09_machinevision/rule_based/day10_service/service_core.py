from datetime import datetime
from pathlib import Path
import csv
import json
import re
import uuid

import cv2
import pytesseract


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

HISTORY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day10"
    / "service_history.json"
)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_labels():
    if not LABEL_PATH.exists():
        raise FileNotFoundError(LABEL_PATH)

    with LABEL_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    return {
        row["file"]: row
        for row in rows
    }


def list_samples():
    if not PRODUCT_DIR.exists():
        raise FileNotFoundError(PRODUCT_DIR)

    return sorted(
        [
            path.name
            for path in PRODUCT_DIR.iterdir()
            if path.suffix.lower()
            in {".png", ".jpg", ".jpeg"}
        ]
    )


def crop_roi(image, roi):
    x1, y1, x2, y2 = map(
        int,
        roi,
    )

    height, width = image.shape[:2]

    if not (
        0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
    ):
        raise ValueError(
            f"ROI가 이미지 범위를 벗어났습니다: {roi}"
        )

    cropped = image[
        y1:y2,
        x1:x2,
    ]

    if cropped.size == 0:
        raise RuntimeError(
            "OCR ROI가 비어 있습니다."
        )

    return cropped


def preprocess_for_ocr(
    roi,
    rules,
):
    mode = rules[
        "preprocess_mode"
    ]

    if mode == "original":
        processed = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB,
        )

    elif mode == "gray":
        processed = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

    elif mode == "clahe":
        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

        clip_limit = float(
            rules.get(
                "clahe_clip_limit",
                2.0,
            )
        )

        grid = tuple(
            rules.get(
                "clahe_grid",
                [8, 8],
            )
        )

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=grid,
        )

        processed = clahe.apply(
            gray
        )

    elif mode == "otsu":
        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

        _, processed = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )

    else:
        raise ValueError(
            f"지원하지 않는 전처리: {mode}"
        )


    processed = cv2.resize(
        processed,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_CUBIC,
    )

    return processed


def run_ocr(
    processed,
    rules,
):
    config = (
        f"--oem {int(rules.get('ocr_oem', 3))} "
        f"--psm {int(rules['ocr_psm'])} "
        f"-c tessedit_char_whitelist="
        f"{rules['whitelist']}"
    )

    raw_text = pytesseract.image_to_string(
        processed,
        lang=rules["ocr_lang"],
        config=config,
    )

    return raw_text


def inspect_sample(file_name):
    rules = load_json(
        OCR_RULE_PATH
    )

    service_config = load_json(
        SERVICE_CONFIG_PATH
    )

    labels = load_labels()


    if file_name not in list_samples():
        raise FileNotFoundError(
            f"등록되지 않은 Sample입니다: {file_name}"
        )

    if file_name not in labels:
        raise KeyError(
            f"Label 정보가 없습니다: {file_name}"
        )

    roi_map = rules[
        "roi_by_file"
    ]

    if file_name not in roi_map:
        raise KeyError(
            f"ROI 정보가 없습니다: {file_name}"
        )


    image_path = (
        PRODUCT_DIR
        / file_name
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            image_path
        )


    roi = crop_roi(
        image,
        roi_map[file_name],
    )

    processed = preprocess_for_ocr(
        roi,
        rules,
    )

    raw_text = run_ocr(
        processed,
        rules,
    )

    ocr_text = (
        raw_text
        .strip()
    )


    pattern_ok = bool(
        re.fullmatch(
            rules["expected_pattern"],
            ocr_text,
        )
    )


    expected_text = (
        labels[file_name][
            "expected_text"
        ]
        .strip()
    )

    expected_match = (
        ocr_text
        == expected_text
    )


    reasons = []

    if not ocr_text:
        reasons.append(
            "NG_OCR_EMPTY"
        )

    if not pattern_ok:
        reasons.append(
            "NG_FORMAT"
        )

    if not expected_match:
        reasons.append(
            "NG_EXPECTED_MISMATCH"
        )


    final_decision = (
        "OK"
        if pattern_ok
        and expected_match
        else "NG"
    )


    result = {
        "request_id":
            uuid.uuid4().hex[:12],

        "inspected_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "service_version":
            service_config[
                "service_version"
            ],

        "inspection_module":
            service_config[
                "inspection_module"
            ],

        "rule_version":
            rules[
                "baseline_version"
            ],

        "file_name":
            file_name,

        "measurements": {
            "ocr_text":
                ocr_text,

            "expected_text":
                expected_text,

            "pattern_ok":
                pattern_ok,

            "expected_match":
                expected_match,
        },

        "final_decision":
            final_decision,

        "reasons":
            reasons,
    }

    return result


def read_history():
    if not HISTORY_PATH.exists():
        return []

    return json.loads(
        HISTORY_PATH.read_text(
            encoding="utf-8"
        )
    )


def append_history(result):
    history = read_history()

    history.append(
        result
    )

    service_config = load_json(
        SERVICE_CONFIG_PATH
    )

    history_limit = int(
        service_config.get(
            "history_limit",
            100,
        )
    )

    history = history[
        -history_limit:
    ]

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_PATH.write_text(
        json.dumps(
            history,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def latest_result():
    history = read_history()

    if not history:
        return None

    return history[-1]


def stats():
    history = read_history()

    ok_count = sum(
        1
        for row in history
        if row["final_decision"]
        == "OK"
    )

    ng_count = sum(
        1
        for row in history
        if row["final_decision"]
        == "NG"
    )

    return {
        "total":
            len(history),

        "ok":
            ok_count,

        "ng":
            ng_count,
    }
