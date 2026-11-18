"""6일차 Mini Challenge — 한 가지 조건만 바꾸기 (Sequence Step 30 → 15) + Random Forest 난수 민감도.

수업 PART 7 규칙
  - 5일차 본수업 18개 Approved Dataset 사용, Test(S03)는 쓰지 않음
  - Train S01 → Validation S02 로만 비교
  - 바꾸는 것은 Sequence Step 하나 (나머지 설정은 configs/day06_baseline.json 그대로)

추가로 확인한 것
  - 교재 Baseline 은 Validation 0.6667 인데 이 PC 결과는 0.5000 → Train 이 6개뿐이라
    Random Forest 난수(random_state)만 바꿔도 결과가 흔들리는지 확인 (Baseline 30 Step 그대로)

출력 (reports/day06/challenge/)
  mini_challenge_predictions.csv   Baseline(30) vs 실험(15) Clip별 예측·확신도
  seed_sensitivity.csv             random_state 0~19 의 Validation 결과
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "day06" / "challenge"
SEQ = ROOT / "data" / "day06" / "sequences"
CHALLENGE_SEQ = ROOT / "data" / "day06" / "challenge_sequences_15"


def load_34():
    spec = importlib.util.spec_from_file_location("build34", ROOT / "scripts" / "34_build_sequence_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def xy(df: pd.DataFrame):
    cols = [c for c in df.columns if len(c) > 4 and c[0] == "t" and c[1:3].isdigit() and c[3] == "_"]
    return df[cols].to_numpy(float), df["action_label"].to_numpy(), cols


def fit_eval(train: pd.DataFrame, val: pd.DataFrame, cfg: dict, seed: int | None = None):
    Xtr, ytr, cols = xy(train)
    Xva, yva, _ = xy(val)
    rf = RandomForestClassifier(n_estimators=int(cfg["n_estimators"]), max_depth=cfg["max_depth"],
                                random_state=int(cfg["random_state"] if seed is None else seed),
                                class_weight="balanced", n_jobs=-1)  # 35번과 같은 설정
    rf.fit(Xtr, ytr)
    proba = rf.predict_proba(Xva)
    pred = rf.classes_[proba.argmax(1)]
    return pred, proba.max(1), len(cols), accuracy_score(yva, pred), f1_score(yva, pred, average="macro", zero_division=0)


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "day06_baseline.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    base_tr, base_va = pd.read_csv(SEQ / "train.csv"), pd.read_csv(SEQ / "val.csv")

    # ── 1) 한 변수 실험: Sequence Step 30 → 15 (34번 코드를 그대로 쓰고 출력 폴더만 바꿈) ──
    m34 = load_34()
    m34.OUTPUT_DIR = CHALLENGE_SEQ
    cfg15 = {**cfg, "sequence_steps": 15}
    for split in ("train", "val"):
        m34.build_split_dataset(split, cfg15)
    ch_tr, ch_va = pd.read_csv(CHALLENGE_SEQ / "train.csv"), pd.read_csv(CHALLENGE_SEQ / "val.csv")

    rows, summary = [], []
    for name, tr, va, c in (("baseline_30", base_tr, base_va, cfg), ("steps_15", ch_tr, ch_va, cfg15)):
        pred, conf, n_feat, acc, f1 = fit_eval(tr, va, c)
        summary.append({"experiment": name, "sequence_steps": c["sequence_steps"], "feature_count": n_feat,
                        "n_estimators": c["n_estimators"], "max_depth": c["max_depth"],
                        "val_accuracy": round(acc, 4), "val_macro_f1": round(f1, 4),
                        "misclassified": int((pred != va["action_label"].to_numpy()).sum())})
        for cid, t, p, cf in zip(va["clip_id"], va["action_label"], pred, conf):
            rows.append({"experiment": name, "clip_id": cid, "true_label": t, "pred_label": p,
                         "correct": int(t == p), "confidence": round(float(cf), 4)})
    S = pd.DataFrame(summary)
    print(S.to_string(index=False))
    P = pd.DataFrame(rows)
    wide = P.pivot(index=["clip_id", "true_label"], columns="experiment", values=["pred_label", "confidence"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(OUT / "mini_challenge_predictions.csv", index=False, encoding="utf-8-sig")
    S.to_csv(OUT / "mini_challenge_summary.csv", index=False, encoding="utf-8-sig")
    print(wide.to_string(index=False))

    # ── 2) 난수 민감도: Baseline(30 Step) 에서 random_state 만 0~19 로 ──
    seeds = []
    for s in range(20):
        pred, _, _, acc, f1 = fit_eval(base_tr, base_va, cfg, seed=s)
        wrong = [f"{t}->{p}" for t, p in zip(base_va["action_label"], pred) if t != p]
        seeds.append({"random_state": s, "val_accuracy": round(acc, 4), "val_macro_f1": round(f1, 4), "wrong": "; ".join(wrong)})
    D = pd.DataFrame(seeds)
    D.to_csv(OUT / "seed_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\nrandom_state 0~19 Validation Accuracy 분포:")
    print(D["val_accuracy"].value_counts().sort_index().to_string())
    print(f"mean {D.val_accuracy.mean():.4f}, min {D.val_accuracy.min():.4f}, max {D.val_accuracy.max():.4f}")


if __name__ == "__main__":
    main()
