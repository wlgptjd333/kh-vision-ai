from pathlib import Path
import json
import sys


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from day10_service.service_core import (
    inspect_sample,
    list_samples,
)


samples = list_samples()

if not samples:
    raise RuntimeError(
        "검사할 Sample이 없습니다."
    )


print("=" * 70)
print("DAY10 SERVICE CORE TEST")
print("=" * 70)


for file_name in samples:

    result = inspect_sample(
        file_name
    )

    print()
    print("File   :", file_name)
    print(
        "OCR    :",
        result[
            "measurements"
        ][
            "ocr_text"
        ],
    )
    print(
        "Expect :",
        result[
            "measurements"
        ][
            "expected_text"
        ],
    )
    print(
        "Final  :",
        result[
            "final_decision"
        ],
    )
    print(
        "Reason :",
        result[
            "reasons"
        ],
    )


first_result = inspect_sample(
    samples[0]
)

output_path = (
    PROJECT_ROOT
    / "reports"
    / "day10"
    / "service_smoke_test.json"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output_path.write_text(
    json.dumps(
        first_result,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print(
    "Saved:",
    output_path,
)
