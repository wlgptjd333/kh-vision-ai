from datetime import datetime
from pathlib import Path
import hashlib
import json
import subprocess


OUTPUT_PATH = Path(
    "reports/day09/version_manifest.json"
)


# --------------------------------------------------
# 1. Dataset / Image Group
# --------------------------------------------------
DATASETS = {
    "day02_preprocess_input":
        Path("samples/day02"),

    "day03_cap_inspection":
        Path("samples/day03"),

    "day04_logo_matching":
        Path("samples/day04"),

    "day05_logo_calibration_test":
        Path("samples/day05/resized"),

    "day06_ocr_product":
        Path("samples/day06/product"),

    "day07_normal_test":
        Path("samples/day07/test/normal"),

    "day08_common_query":
        Path("samples/day08/common_query"),
}


# --------------------------------------------------
# 2. Config
# --------------------------------------------------
CONFIG_FILES = [
    Path("configs/day05_logo_rules.json"),
    Path("configs/day06_ocr_rules.json"),
    Path("configs/day06_labels.csv"),
    Path("configs/day07_anomaly_rules.json"),
    Path("configs/day08_one_shot_rules.json"),
    Path("configs/day08_fewshot_config.json"),
]


# --------------------------------------------------
# 3. Artifact
# --------------------------------------------------
ARTIFACT_FILES = [
    Path("artifacts/day07/normal_reference.npz"),
    Path("artifacts/day08/one_shot_reference.npy"),
    Path("artifacts/day08/fewshot_prototypes.npz"),
]


# --------------------------------------------------
# 4. Result Source
# --------------------------------------------------
RESULT_SOURCE_FILES = [
    # Day01~04
    Path("reports/day01_problem_definition.md"),
    Path("reports/day02_parameter_table.csv"),
    Path("reports/day02_preprocess_reason.md"),
    Path("reports/day03_inspection_results.csv"),
    Path("reports/day04_homography_results.csv"),

    # Day05
    Path("reports/day05_calibration_measurements.csv"),
    Path("reports/day05_test_results.csv"),

    # Day06
    Path("reports/day06_ocr_results.csv"),
    Path("reports/day06_ocr_metrics.md"),

    # Day07
    Path("reports/day07_anomaly_scores.csv"),
    Path("reports/day07_test_metrics.md"),

    # Day08
    Path("reports/day08_one_shot_results.csv"),
    Path("reports/day08_fewshot_results.csv"),
    Path("reports/day08_method_comparison.csv"),
]


def file_sha256(path):
    sha = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


def directory_summary(directory):
    if not directory.exists():
        return {
            "exists": False,
            "file_count": 0,
            "fingerprint": None,
        }

    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
    )

    records = []

    for path in files:
        relative = path.relative_to(
            directory
        )

        records.append(
            f"{relative.as_posix()}|"
            f"{path.stat().st_size}"
        )

    joined = "\n".join(
        records
    )

    fingerprint = hashlib.sha256(
        joined.encode("utf-8")
    ).hexdigest()

    return {
        "exists": True,
        "file_count": len(files),
        "fingerprint": fingerprint,
    }


def file_summary(path):
    return {
        "exists": path.exists(),
        "size":
            path.stat().st_size
            if path.exists()
            else None,
        "sha256":
            file_sha256(path)
            if path.exists()
            else None,
    }


try:
    git_commit = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()

except Exception:
    git_commit = "NO_GIT"


manifest = {
    "created_at":
        datetime.now().isoformat(
            timespec="seconds"
        ),

    "git_commit":
        git_commit,

    "datasets": {},
    "configs": {},
    "artifacts": {},
    "result_sources": {},
}


for name, directory in DATASETS.items():
    manifest["datasets"][name] = (
        directory_summary(
            directory
        )
    )


for path in CONFIG_FILES:
    manifest["configs"][
        str(path)
    ] = file_summary(
        path
    )


for path in ARTIFACT_FILES:
    manifest["artifacts"][
        str(path)
    ] = file_summary(
        path
    )


for path in RESULT_SOURCE_FILES:
    manifest["result_sources"][
        str(path)
    ] = file_summary(
        path
    )


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print("=" * 70)
print("DAY09 VERSION MANIFEST")
print("=" * 70)

print(
    "Git Commit:",
    git_commit,
)

for name, info in (
    manifest["datasets"].items()
):
    print(
        "Dataset:",
        name,
        "exists=",
        info["exists"],
        "files=",
        info["file_count"],
    )

print(
    "Saved:",
    OUTPUT_PATH,
)
