from __future__ import annotations

# ==================================================
# team03_05_summarize_sequences.py
# 3팀 Sequence CSV 136개 → 영상별 QA 요약표 (4일차 STEP 5)
# ==================================================
# 4일차 23번(Sequence 품질 요약)을 clip_id 단위로 확장.
# 자동으로는 PASS / REVIEW 만 붙인다. RESHOOT · EXCLUDE 는 사람이 영상을 보고 정한다.
# 기준값(아래 상수)은 3팀 영상 표본을 보고 정한 것이며 모든 팀에 맞는 값은 아니다.
#
# 입력: reports/team_projects/team03/sequences/<clip_id>_sequence.csv  (team03_03 결과)
# 출력: reports/team_projects/team03/tables/sequence_qa.csv
# 실행: python scripts/team03_05_summarize_sequences.py

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "team03_clip_manifest.csv"
SEQ_DIR = ROOT / "reports" / "team_projects" / "team03" / "sequences"
OUT = ROOT / "reports" / "team_projects" / "team03" / "tables" / "sequence_qa.csv"

KP_CONF = 0.50            # 관절 Confidence 기준 (project.json 과 같은 값)
EDGE_MARGIN = 0.01        # 사람 상자가 화면 끝 1% 안에 닿으면 "잘림 의심"

# REVIEW 로 표시하는 기준: (열 이름, 비교, 값, 사유 이름)
RULES = [
    ("pose_ok_ratio", "<", 0.98, "pose_miss"),            # 사람을 놓친 Frame 이 2% 넘음
    ("longest_gap_sec", ">=", 0.20, "long_gap"),          # 0.2초 이상 연속으로 놓침
    ("core_low_ratio", ">", 0.05, "core_low"),            # 어깨·골반(정규화 기준) 저신뢰
    ("lower_low_ratio", ">", 0.10, "lower_low"),          # 무릎·발목 저신뢰
    ("wrist_low_ratio", ">", 0.40, "wrist_low"),          # 손목 저신뢰 (측면에서 반대쪽 손이 가려짐)
    ("multi_person_ratio", ">", 0.05, "multi_person"),    # 다른 사람이 같이 잡힘
    ("edge_touch_ratio", ">", 0.20, "edge_touch"),        # 몸이 화면 끝에 닿음 (너무 가까움)
    ("body_ok_ratio", "<", 0.98, "body_norm_fail"),       # 몸 크기 정규화 실패
]


def longest_run(flags) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def summarize(clip: dict, path: Path) -> dict:
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    n = len(d)
    ok = d[d.pose_ok == 1]
    fps = float(d.fps.iloc[0]) if n else 0.0
    w, h = (int(d.img_w.iloc[0]), int(d.img_h.iloc[0])) if n else (0, 0)

    def low(joints):                                   # Pose 성공 Frame 중 해당 관절 하나라도 저신뢰인 비율
        if ok.empty:
            return float("nan")
        return float((ok[[f"raw_{j}_c" for j in joints]].min(axis=1) < KP_CONF).mean())

    edge = ((ok.box_x1 < w * EDGE_MARGIN) | (ok.box_y1 < h * EDGE_MARGIN)
            | (ok.box_x2 > w * (1 - EDGE_MARGIN)) | (ok.box_y2 > h * (1 - EDGE_MARGIN)))
    gap = longest_run((d.pose_ok == 0).tolist())
    src = ok.body_scale_src.value_counts(normalize=True) if not ok.empty else {}
    row = {
        **{k: clip[k] for k in ("clip_id", "subject_id", "action_id", "action_label", "view")},
        "frames": n, "fps": round(fps, 2), "duration_sec": round(n / fps, 2) if fps else "",
        "orientation": "portrait" if h > w else "landscape",
        "pose_ok_ratio": round(float(d.pose_ok.mean()), 4),
        "feature_ok_ratio": round(float(d.feature_ok.mean()), 4),
        "body_ok_ratio": round(float(d.body_feature_ok.mean()), 4),
        "mean_conf": round(float(ok.mean_keypoint_conf.mean()), 4),
        "core_low_ratio": round(float((ok.core_ok == 0).mean()), 4),
        "lower_low_ratio": round(low(["left_knee", "right_knee", "left_ankle", "right_ankle"]), 4),
        "wrist_low_ratio": round(low(["left_wrist", "right_wrist"]), 4),
        "scale_warn_ratio": round(float((ok.scale_warn == 1).mean()), 4),      # 어깨 폭 정규화(38-D)를 믿기 어려운 비율
        "shoulder_scale_ratio": round(float(src.get("shoulder", 0.0)), 4),     # 몸 크기 기준으로 어깨 폭을 쓴 비율
        "multi_person_ratio": round(float((d.people_count > 1).mean()), 4),
        "edge_touch_ratio": round(float(edge.mean()), 4),
        "box_h_median": round(float(((ok.box_y2 - ok.box_y1) / h).median()), 3),
        "longest_gap_frames": gap, "longest_gap_sec": round(gap / fps, 2) if fps else "",
        "tilt_max": round(float(ok.torso_tilt_deg.max()), 1),
    }
    reasons = []
    for col, op, limit, name in RULES:
        v = row[col]
        if v == "" or (isinstance(v, float) and np.isnan(v)):
            continue
        if (op == "<" and v < limit) or (op == ">" and v > limit) or (op == ">=" and v >= limit):
            reasons.append(name)
    row["auto_status"] = "REVIEW" if reasons else "PASS"
    row["auto_reasons"] = "|".join(reasons)
    return row


def main() -> None:
    with MANIFEST.open(encoding="utf-8-sig") as f:
        clips = list(csv.DictReader(f))
    rows, missing = [], []
    for clip in clips:
        path = SEQ_DIR / f"{clip['clip_id']}_sequence.csv"
        if not path.exists():
            missing.append(clip["clip_id"])
            continue
        rows.append(summarize(clip, path))
    if not rows:
        sys.exit("Sequence CSV 가 없습니다. 먼저 team03_03_build_sequences.py --all 을 실행하세요.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    qa = pd.DataFrame(rows)
    qa.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"영상 {len(qa)}개 요약 (Sequence 없음 {len(missing)}개) · 전체 Frame {int(qa.frames.sum()):,}")
    print(f"자동 판정: {qa.auto_status.value_counts().to_dict()}")
    print(f"Pose 성공 평균 {qa.pose_ok_ratio.mean():.1%} · 38-D 성공 {qa.feature_ok_ratio.mean():.1%}"
          f" · 몸 크기 정규화 성공 {qa.body_ok_ratio.mean():.1%}")
    by_view = qa.groupby("view")[["mean_conf", "wrist_low_ratio", "scale_warn_ratio"]].mean().round(3)
    for view, r in by_view.iterrows():
        print(f"  {'정면' if view == 'F' else '측면'}: 평균 Conf {r.mean_conf} · 손목 저신뢰 {r.wrist_low_ratio:.1%}"
              f" · 어깨 폭 경고 {r.scale_warn_ratio:.1%}")
    review = qa[qa.auto_status == "REVIEW"]
    if len(review):
        print("\nREVIEW 대상:")
        for _, r in review.iterrows():
            print(f"  {r.clip_id}  {r.auto_reasons}")
    if missing:
        print(f"\nSequence 가 없는 clip: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    print(f"\nSaved: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
