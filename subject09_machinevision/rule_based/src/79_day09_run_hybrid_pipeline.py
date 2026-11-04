from pathlib import Path
import json
import pandas as pd

from day09_hybrid_core import hybrid_judge, calculate_metrics

INPUT_PATH = Path("reports/day09/hybrid_inputs.csv")
POLICY_PATH = Path("configs/day09/hybrid_policy.json")
RESULT_PATH = Path("reports/day09/hybrid_results.csv")
METRIC_PATH = Path("reports/day09/hybrid_metrics.csv")
REPORT_PATH = Path("reports/day09/hybrid_comparison.md")

if not INPUT_PATH.exists() or not POLICY_PATH.exists():
    raise FileNotFoundError("먼저 78_day09_make_hybrid_demo.py를 실행하세요.")

inputs = pd.read_csv(INPUT_PATH, dtype={"sample_id": str})
policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
required = {"sample_id", "file", "actual", "one_shot_predicted", "few_shot_predicted"}
if required - set(inputs.columns):
    raise RuntimeError(f"Hybrid 입력 CSV 컬럼이 부족합니다: {sorted(required - set(inputs.columns))}")
if inputs["sample_id"].duplicated().any() or len(inputs) != 20:
    raise RuntimeError("공통 20장이어야 하며 sample_id 중복이 없어야 합니다.")

results = []
for _, row in inputs.iterrows():
    actual = row["actual"]
    one = row["one_shot_predicted"]
    few = row["few_shot_predicted"]
    for mode in policy["policies"]:
        result = hybrid_judge(one, few, mode)
        results.append({
            "sample_id": row["sample_id"],
            "file": row["file"],
            "actual": actual,
            "policy": mode,
            "one_shot_predicted": one,
            "few_shot_predicted": few,
            "hybrid_predicted": result["predicted"],
            "correct": actual == result["predicted"],
            "reason": result["reason"],
            "policy_version": policy["policy_version"],
        })

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

methods = {
    "one_shot": inputs["one_shot_predicted"].tolist(),
    "few_shot": inputs["few_shot_predicted"].tolist(),
}
for mode in policy["policies"]:
    methods[f"hybrid_{mode.lower()}"] = [
        hybrid_judge(o, f, mode)["predicted"]
        for o, f in zip(inputs["one_shot_predicted"], inputs["few_shot_predicted"])
    ]

metric_rows = []
for method, predictions in methods.items():
    metric_rows.append({
        "method": method,
        **calculate_metrics(inputs["actual"].tolist(), predictions),
    })
metrics = pd.DataFrame(metric_rows)
metrics.to_csv(METRIC_PATH, index=False, encoding="utf-8-sig")

# Day08 원래 Accuracy와 단독검사 결과가 일치하는지 재확인합니다.
original = pd.read_csv("reports/day08_method_comparison.csv")
if not {"method", "accuracy"}.issubset(original.columns):
    raise RuntimeError("Day08 Method Comparison 형식이 다릅니다.")
for method in ["one_shot", "few_shot"]:
    stored = original.loc[original["method"] == method, "accuracy"]
    current = metrics.loc[metrics["method"] == method, "accuracy"].iloc[0]
    if len(stored) != 1 or round(float(stored.iloc[0]), 4) != round(float(current), 4):
        raise RuntimeError(f"{method} 저장 Accuracy와 다시 계산한 값이 다릅니다.")

lines = [
    "# Day09 Hybrid 성능 비교", "",
    "- 데이터: Day08 MVTec bottle Common Query 20장",
    "- 정책: OR / AND (Test 결과를 보기 전에 고정한 비교 정책)",
    "- 두 검사 모두 기존 Day08 예측 CSV 사용; 재학습·재추론 없음", "",
    "| 검사 방식 | Accuracy | Precision | Recall | FP | FN |",
    "|---|---:|---:|---:|---:|---:|",
]
for m in metric_rows:
    lines.append(
        f"| {m['method']} | {m['accuracy']:.1%} | "
        f"{m['anomaly_precision']:.1%} | {m['anomaly_recall']:.1%} | "
        f"{m['fp']} | {m['fn']} |"
    )
lines += [
    "", "## 관찰", "",
    "- OR에서 FN이 줄었나요? FP는 늘었나요?", 
    "- AND에서 FP가 줄었나요? FN은 늘었나요?",
    "- 어떤 상황에 어떤 정책을 선택할지 근거를 적으세요.",
    "- 이 20장은 비교용 Test이므로 여기서 정책을 선택했다면 **별도 최종 Test**가 필요합니다.",
]
REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("=" * 65)
print("DAY09 HYBRID COMPARISON")
print("=" * 65)
print(metrics.to_string(index=False))
print("Saved:", RESULT_PATH)
print("Saved:", METRIC_PATH)
print("Saved:", REPORT_PATH)
