from pathlib import Path
import json
import pandas as pd

ONE_PATH = Path("reports/day08_one_shot_results.csv")
FEW_PATH = Path("reports/day08_fewshot_results.csv")
AUDIT_PATH = Path("reports/day09/day08_reproducibility_audit.json")
OUTPUT_DIR = Path("reports/day09")
POLICY_PATH = Path("configs/day09/hybrid_policy.json")
INPUT_PATH = OUTPUT_DIR / "hybrid_inputs.csv"

# PART3에서 같은 Query와 결과를 이미 검증했는지 확인합니다.
if not AUDIT_PATH.exists():
    raise FileNotFoundError("PART3의 Day08 Audit를 먼저 실행하세요.")

audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
if not audit.get("same_common_query") or not audit.get("all_metric_match"):
    raise RuntimeError("Day08 Audit가 통과하지 않았습니다.")


def load_result(path, prefix):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"sample_id": str})
    required = {"sample_id", "file", "actual", "predicted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: 누락 컬럼 {sorted(missing)}")

    df = df[["sample_id", "file", "actual", "predicted"]].copy()
    for col in ["sample_id", "file", "actual", "predicted"]:
        if df[col].isna().any():
            raise ValueError(f"{path}: {col} 빈 값")
        df[col] = df[col].astype(str).str.strip()
        if df[col].eq("").any():
            raise ValueError(f"{path}: {col} 빈 문자열")

    if df["sample_id"].duplicated().any():
        raise ValueError(f"{path}: sample_id 중복")
    valid = {"Normal", "Anomaly"}
    if not set(df["actual"]).issubset(valid):
        raise ValueError(f"{path}: actual에는 Normal/Anomaly만 사용합니다.")
    if not set(df["predicted"]).issubset(valid):
        raise ValueError(f"{path}: predicted에는 Normal/Anomaly만 사용합니다.")

    return df.rename(columns={
        "file": f"{prefix}_file",
        "actual": f"{prefix}_actual",
        "predicted": f"{prefix}_predicted",
    })


one = load_result(ONE_PATH, "one_shot")
few = load_result(FEW_PATH, "few_shot")

if set(one["sample_id"]) != set(few["sample_id"]):
    raise RuntimeError("One-shot/Few-shot의 검사 이미지 ID가 다릅니다.")

merged = one.merge(few, on="sample_id", how="inner", validate="one_to_one")
if not merged["one_shot_actual"].eq(merged["few_shot_actual"]).all():
    raise RuntimeError("같은 이미지의 실제 정답이 서로 다릅니다.")
if not merged["one_shot_file"].eq(merged["few_shot_file"]).all():
    raise RuntimeError("같은 ID인데 이미지 경로가 다릅니다. Day08 결과를 확인하세요.")

if len(merged) != 20:
    raise RuntimeError(f"이번 수업은 공통 이미지 20장을 사용합니다. 현재 {len(merged)}장")

merged = merged.sort_values("sample_id").reset_index(drop=True)
inputs = pd.DataFrame({
    "sample_id": merged["sample_id"],
    "file": merged["one_shot_file"],
    "actual": merged["one_shot_actual"],
    "one_shot_predicted": merged["one_shot_predicted"],
    "few_shot_predicted": merged["few_shot_predicted"],
})

policy = {
    "policy_version": "day09-hybrid-v2.0",
    "dataset": "Day08 MVTec bottle common_query 20",
    "positive_label": "Anomaly",
    "policies": ["OR", "AND"],  # 결과 확인 전에 고정한 비교 정책
    "sources": {
        "one_shot": str(ONE_PATH),
        "few_shot": str(FEW_PATH),
        "day08_audit": str(AUDIT_PATH),
    },
    "version_manifest": "reports/day09/version_manifest.json",
    "note": "실제 Day08 저장 예측의 결합; 새로운 CNN 학습은 수행하지 않음",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
inputs.to_csv(INPUT_PATH, index=False, encoding="utf-8-sig")
POLICY_PATH.write_text(
    json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Common Query:", len(inputs))
print("Saved:", INPUT_PATH)
print("Policy:", POLICY_PATH)
