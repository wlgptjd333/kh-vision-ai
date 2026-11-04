from pathlib import Path
import json

import httpx


BASE_URL = (
    "http://127.0.0.1:8000"
)


with httpx.Client(
    timeout=30.0
) as client:

    health = client.get(
        f"{BASE_URL}/api/health"
    )

    health.raise_for_status()


    sample_response = client.get(
        f"{BASE_URL}/api/samples"
    )

    sample_response.raise_for_status()

    samples = sample_response.json()[
        "samples"
    ]


    if not samples:
        raise RuntimeError(
            "검사 가능한 Sample이 없습니다."
        )


    inspect_response = client.post(
        f"{BASE_URL}/api/inspect",
        json={
            "file_name":
                samples[0]
        },
    )

    inspect_response.raise_for_status()

    result = (
        inspect_response
        .json()
    )


print("=" * 70)
print("DAY10 API TEST")
print("=" * 70)

print(
    "Health:",
    health.json()
)

print(
    "Sample:",
    samples[0]
)

print(
    "Final:",
    result[
        "final_decision"
    ]
)

print(
    "Reason:",
    result[
        "reasons"
    ]
)


output_path = Path(
    "reports/day10/api_test_result.json"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output_path.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(
    "Saved:",
    output_path
)
