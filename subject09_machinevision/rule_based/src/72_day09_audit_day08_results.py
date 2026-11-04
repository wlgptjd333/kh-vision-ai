from pathlib import Path
import csv
import hashlib
import json

import pandas as pd


ONE_RESULT = Path(
    "reports/day08_one_shot_results.csv"
)

FEW_RESULT = Path(
    "reports/day08_fewshot_results.csv"
)

COMPARISON = Path(
    "reports/day08_method_comparison.csv"
)


ONE_CONFIG = Path(
    "configs/day08_one_shot_rules.json"
)

FEW_CONFIG = Path(
    "configs/day08_fewshot_config.json"
)

ONE_ARTIFACT = Path(
    "artifacts/day08/one_shot_reference.npy"
)

FEW_ARTIFACT = Path(
    "artifacts/day08/fewshot_prototypes.npz"
)


AUDIT_CSV = Path(
    "reports/day09/"
    "day08_reproducibility_audit.csv"
)

AUDIT_JSON = Path(
    "reports/day09/"
    "day08_reproducibility_audit.json"
)


REQUIRED_RESULT_COLUMNS = {
    "sample_id",
    "actual",
    "predicted",
}


REQUIRED_COMPARISON_COLUMNS = {
    "method",
    "samples",
    "accuracy",
    "anomaly_precision",
    "anomaly_recall",
    "tp",
    "tn",
    "fp",
    "fn",
}


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def validate_result_schema(
    df,
    name,
):
    missing = (
        REQUIRED_RESULT_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{name} Result 컬럼이 부족합니다: "
            f"{sorted(missing)}"
        )

    if df["sample_id"].duplicated().any():
        duplicated = (
            df.loc[
                df["sample_id"].duplicated(),
                "sample_id",
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            f"{name} sample_id 중복: "
            f"{duplicated}"
        )


def metrics(df):
    actual = (
        df["actual"]
        .astype(str)
        .str.strip()
    )

    predicted = (
        df["predicted"]
        .astype(str)
        .str.strip()
    )

    tp = int(
        (
            (actual == "Anomaly")
            & (predicted == "Anomaly")
        ).sum()
    )

    tn = int(
        (
            (actual == "Normal")
            & (predicted == "Normal")
        ).sum()
    )

    fp = int(
        (
            (actual == "Normal")
            & (predicted == "Anomaly")
        ).sum()
    )

    fn = int(
        (
            (actual == "Anomaly")
            & (predicted == "Normal")
        ).sum()
    )

    total = (
        tp + tn + fp + fn
    )

    if total != len(df):
        raise RuntimeError(
            "actual / predicted 값에 "
            "Normal 또는 Anomaly가 아닌 값이 있습니다."
        )

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    return {
        "samples": total,
        "accuracy": accuracy,
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# --------------------------------------------------
# 1. Result 읽기
# --------------------------------------------------
one = pd.read_csv(
    ONE_RESULT
)

few = pd.read_csv(
    FEW_RESULT
)

comparison = pd.read_csv(
    COMPARISON
)


validate_result_schema(
    one,
    "One-shot",
)

validate_result_schema(
    few,
    "Few-shot",
)


missing = (
    REQUIRED_COMPARISON_COLUMNS
    - set(comparison.columns)
)

if missing:
    raise RuntimeError(
        "Method Comparison 컬럼이 부족합니다: "
        f"{sorted(missing)}"
    )


# --------------------------------------------------
# 2. 같은 Common Query인지 확인
# --------------------------------------------------
one_ids = set(
    one["sample_id"].astype(str)
)

few_ids = set(
    few["sample_id"].astype(str)
)

same_query = (
    one_ids == few_ids
)

if not same_query:
    only_one = sorted(
        one_ids - few_ids
    )

    only_few = sorted(
        few_ids - one_ids
    )

    raise RuntimeError(
        "One-shot과 Few-shot Query가 다릅니다.\n"
        f"Only One-shot: {only_one}\n"
        f"Only Few-shot: {only_few}"
    )


# --------------------------------------------------
# 3. Metric 재계산
# --------------------------------------------------
recomputed = {
    "one_shot":
        metrics(one),

    "few_shot":
        metrics(few),
}


# --------------------------------------------------
# 4. 저장된 Comparison과 비교
# --------------------------------------------------
comparison = comparison.set_index(
    "method"
)

rows = []
all_metric_match = True

for method in [
    "one_shot",
    "few_shot",
]:
    if method not in comparison.index:
        raise RuntimeError(
            f"Method Comparison에 {method} 행이 없습니다."
        )

    current = recomputed[
        method
    ]

    stored = comparison.loc[
        method
    ]

    checks = {
        "samples_match":
            int(stored["samples"])
            == current["samples"],

        "accuracy_match":
            round(
                float(stored["accuracy"]),
                4,
            )
            == round(
                current["accuracy"],
                4,
            ),

        "precision_match":
            round(
                float(
                    stored[
                        "anomaly_precision"
                    ]
                ),
                4,
            )
            == round(
                current[
                    "anomaly_precision"
                ],
                4,
            ),

        "recall_match":
            round(
                float(
                    stored[
                        "anomaly_recall"
                    ]
                ),
                4,
            )
            == round(
                current[
                    "anomaly_recall"
                ],
                4,
            ),

        "tp_match":
            int(stored["tp"])
            == current["tp"],

        "tn_match":
            int(stored["tn"])
            == current["tn"],

        "fp_match":
            int(stored["fp"])
            == current["fp"],

        "fn_match":
            int(stored["fn"])
            == current["fn"],
    }

    metric_match = all(
        checks.values()
    )

    all_metric_match = (
        all_metric_match
        and metric_match
    )

    rows.append(
        {
            "method":
                method,

            "samples":
                current["samples"],

            "accuracy":
                round(
                    current["accuracy"],
                    4,
                ),

            "anomaly_precision":
                round(
                    current[
                        "anomaly_precision"
                    ],
                    4,
                ),

            "anomaly_recall":
                round(
                    current[
                        "anomaly_recall"
                    ],
                    4,
                ),

            "tp":
                current["tp"],

            "tn":
                current["tn"],

            "fp":
                current["fp"],

            "fn":
                current["fn"],

            "same_common_query":
                same_query,

            "stored_metrics_match":
                metric_match,
        }
    )


# --------------------------------------------------
# 5. Audit CSV 저장
# --------------------------------------------------
AUDIT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with AUDIT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


# --------------------------------------------------
# 6. Config / Artifact / Result Hash 기록
# --------------------------------------------------
files_for_hash = [
    ONE_CONFIG,
    FEW_CONFIG,
    ONE_ARTIFACT,
    FEW_ARTIFACT,
    ONE_RESULT,
    FEW_RESULT,
    COMPARISON,
]

hashes = {}

for path in files_for_hash:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    hashes[str(path)] = sha256(
        path
    )


audit = {
    "same_common_query":
        same_query,

    "query_count":
        len(one_ids),

    "all_metric_match":
        all_metric_match,

    "metrics":
        recomputed,

    "sha256":
        hashes,
}


AUDIT_JSON.write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print("=" * 70)
print("DAY08 REPRODUCIBILITY AUDIT")
print("=" * 70)

print(
    "Same Common Query:",
    same_query,
)

print(
    "Query Count:",
    len(one_ids),
)

for row in rows:
    print()
    print(
        "Method   :",
        row["method"],
    )

    print(
        "Accuracy :",
        row["accuracy"],
    )

    print(
        "TP/TN/FP/FN:",
        row["tp"],
        row["tn"],
        row["fp"],
        row["fn"],
    )

    print(
        "Stored Metric Match:",
        row[
            "stored_metrics_match"
        ],
    )

print()
print(
    "All Metric Match:",
    all_metric_match,
)

print(
    "Audit CSV :",
    AUDIT_CSV,
)

print(
    "Audit JSON:",
    AUDIT_JSON,
)

if not all_metric_match:
    raise RuntimeError(
        "Day08 저장 Metric과 재계산 Metric이 다릅니다."
    )
