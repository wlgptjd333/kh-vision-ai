from __future__ import annotations

# ==================================================
# team03_03_build_sequences.py
# 3팀 MP4 → Frame 마다 Pose → 정규화 38-D + 3팀 Feature → <clip_id>_sequence.csv (4일차 STEP 4)
# ==================================================
# 3일차 18번(영상 → 38-D CSV)을 바탕으로 4일차 교재의 확장 사항을 더함:
#   - 출력 이름을 clip_id 로 → 같은 동작 여러 영상이 덮어써지지 않음
#   - 출처(clip_id·subject·action·view), 시간(frame_id·time_sec·fps), 화면 크기
#   - 품질: pose_ok / feature_ok / fail_reason / 사람 수 / 관절별 Confidence / 기준 관절(어깨·골반) 상태
#   - 원본 Keypoint 17×(x, y, conf) 를 따로 저장 (정규화로 위치 정보가 사라지지 않게)
#   - 3팀 Feature: 상체 기울기·몸 상자 비율·골반 화면 위치 등 (src/team_feature_utils.py)
#   - 몸 크기 정규화 좌표 bn_* 34개: 어깨 폭 정규화(38-D)가 측면·누운 자세에서 튀는 문제를 보완
#   - 사람 상자(box_x1~y2): 화면 끝에 몸이 잘렸는지 QA 에서 확인
#   - Confidence 가 낮다고 Frame 을 버리지 않고 상태만 기록
#
# 실행 (교재 순서: 1개 → 2~3개 → 같은 동작 여러 개 → 17개 대표 → 전체):
#   python scripts/team03_03_build_sequences.py --clip T03_S04_A01_T01_F
#   python scripts/team03_03_build_sequences.py --clip T03_S04_A01_T01_F --save-video
#   python scripts/team03_03_build_sequences.py --all
# 출력: reports/team_projects/team03/sequences/<clip_id>_sequence.csv
#       (--save-video) reports/team_projects/team03/videos/<clip_id>_skeleton.mp4

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config, resolve_device  # noqa: E402
from src.feature_utils import (LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER,  # noqa: E402
                               normalize_by_shoulder_width, selected_joint_angles)
from src.team_feature_utils import (SCALE_WARN_RATIO, TEAM03_FEATURE_NAMES,  # noqa: E402
                                    normalize_by_body_scale, team03_features)

# 16번의 Feature 이름(38개)·관절 이름을 그대로 재사용
_spec = importlib.util.spec_from_file_location("frame_feature16", ROOT / "scripts" / "16_make_frame_feature.py")
_m16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m16)
FEATURE_NAMES = _m16.make_feature_names()
KP_NAMES = _m16.COCO_KEYPOINT_NAMES
ANGLE_NAMES = ["left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"]
CORE = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]       # 정규화 기준 관절

MANIFEST = ROOT / "configs" / "team03_clip_manifest.csv"
VIDEOS = ROOT / "data" / "team_projects" / "team03" / "rgb" / "videos"
OUT_DIR = ROOT / "reports" / "team_projects" / "team03"

META = ["clip_id", "subject_id", "action_id", "action_label", "view", "source_type",
        "frame_id", "time_sec", "fps", "img_w", "img_h",
        "pose_ok", "feature_ok", "fail_reason", "people_count", "box_conf",
        "box_x1", "box_y1", "box_x2", "box_y2",
        "mean_keypoint_conf", "low_keypoint_count", "core_ok", "scale_warn",
        "hip_center_x", "hip_center_y", "shoulder_width",
        "body_feature_ok", "body_scale", "body_scale_src"]
RAW = [f"raw_{n}_{k}" for n in KP_NAMES for k in ("x", "y", "c")]
# 몸 크기(body scale) 정규화 좌표 34개: 측면·누운 자세에서도 값이 튀지 않음 (38-D 는 수업 형식 그대로 둠)
BODY_NAMES = [f"bn_{n}_{k}" for n in KP_NAMES for k in ("x", "y")]
COLUMNS = META + FEATURE_NAMES + BODY_NAMES + TEAM03_FEATURE_NAMES + RAW


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", action="append", default=[], help="clip_id (여러 번 줄 수 있음)")
    p.add_argument("--all", action="store_true", help="Manifest 의 모든 영상")
    p.add_argument("--max-frames", type=int, default=0, help="영상마다 최대 Frame 수 (0 = 전체)")
    p.add_argument("--kp-conf", type=float, default=None, help="저신뢰 관절 기준 (기본: project.json)")
    p.add_argument("--save-video", action="store_true", help="Skeleton 을 그린 확인용 영상 저장")
    return p.parse_args()


def frame_row(result, img_w, img_h, kp_conf):
    """YOLO 결과 1개 → CSV 한 행에 들어갈 품질·Feature·원본 값 (첫 번째 = 가장 확실한 사람)"""
    row = {"pose_ok": 0, "feature_ok": 0, "body_feature_ok": 0, "fail_reason": "", "people_count": 0}
    kps = result.keypoints
    if kps is None or kps.xy is None or len(kps.xy) == 0:
        row["fail_reason"] = "no_person"
        return row

    row["people_count"] = len(kps.xy)
    if result.boxes is not None and len(result.boxes):
        row["box_conf"] = round(float(result.boxes.conf[0]), 4)
        for name, v in zip(("box_x1", "box_y1", "box_x2", "box_y2"), result.boxes.xyxy[0].cpu().numpy()):
            row[name] = round(float(v), 1)
    xy = kps.xy[0].cpu().numpy()
    if kps.conf is not None:
        conf = kps.conf[0].cpu().numpy()
    else:                                   # Confidence 가 없으면 1.0 으로 꾸미지 않고 비워 둠
        conf = np.full(17, np.nan, dtype=np.float32)
    row["pose_ok"] = 1
    row["mean_keypoint_conf"] = round(float(np.nanmean(conf)), 4)
    row["low_keypoint_count"] = int(np.sum(conf < kp_conf))
    row["core_ok"] = int(bool(np.all(conf[CORE] >= kp_conf)))
    for i, n in enumerate(KP_NAMES):
        row[f"raw_{n}_x"], row[f"raw_{n}_y"] = round(float(xy[i, 0]), 2), round(float(xy[i, 1]), 2)
        row[f"raw_{n}_c"] = round(float(conf[i]), 4)

    team = team03_features(xy, conf, img_w, img_h, kp_conf)
    row.update(team)
    ratio = team["shoulder_ratio"]
    row["scale_warn"] = int(ratio != "" and ratio < SCALE_WARN_RATIO)

    try:                                    # 몸 크기 정규화 (38-D 성공 여부와 따로 기록)
        body_norm, b_scale, b_src = normalize_by_body_scale(xy)
        if np.all(np.isfinite(body_norm)):
            row.update(body_feature_ok=1, body_scale=round(b_scale, 2), body_scale_src=b_src)
            for i, n in enumerate(KP_NAMES):
                row[f"bn_{n}_x"], row[f"bn_{n}_y"] = round(float(body_norm[i, 0]), 6), round(float(body_norm[i, 1]), 6)
    except ValueError:
        pass

    try:
        normalized, center, scale = normalize_by_shoulder_width(xy)
        angles = selected_joint_angles(xy)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        row["fail_reason"] = "shoulder_width_zero"
        return row
    values = [float(v) for point in normalized for v in point[:2]] + [float(angles[n]) for n in ANGLE_NAMES]
    if len(values) != 38 or not np.all(np.isfinite(values)):
        row["fail_reason"] = "nan_feature"
        return row
    row["feature_ok"] = 1
    row.update({"hip_center_x": round(float(center[0]), 2), "hip_center_y": round(float(center[1]), 2),
                "shoulder_width": round(float(scale), 2)})
    for name, v in zip(FEATURE_NAMES, values):
        row[name] = round(v, 6)
    return row


def build(clip, model, cfg, device, kp_conf, max_frames, save_video):
    path = VIDEOS / clip["video_file"]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer, rows, n = None, [], 0
    while max_frames <= 0 or n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        h, w = frame.shape[:2]
        result = model.predict(source=frame, imgsz=cfg["image_size"], conf=cfg["person_confidence"],
                               device=device, verbose=False)[0]
        row = {c: "" for c in COLUMNS}
        row.update({k: clip[k] for k in ("clip_id", "subject_id", "action_id", "action_label", "view",
                                          "source_type")})
        row.update(frame_id=n, time_sec=round((n - 1) / fps, 3), fps=round(fps, 3), img_w=w, img_h=h)
        row.update(frame_row(result, w, h, kp_conf))
        rows.append(row)
        if save_video:
            if writer is None:
                vdir = OUT_DIR / "videos"
                vdir.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(vdir / f"{clip['clip_id']}_skeleton.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            img = result.plot(boxes=False)
            tilt = row["torso_tilt_deg"]
            text = f"{n}  conf {row['mean_keypoint_conf']}  tilt {tilt}  {'OK' if row['feature_ok'] else row['fail_reason']}"
            cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            writer.write(img)
    cap.release()
    if writer:
        writer.release()

    out = OUT_DIR / "sequences" / f"{clip['clip_id']}_sequence.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    return out, rows


def summarize(rows):
    n = len(rows)
    pose = sum(r["pose_ok"] for r in rows)
    feat = sum(r["feature_ok"] for r in rows)
    body = sum(r["body_feature_ok"] for r in rows)
    core = sum(1 for r in rows if r["core_ok"] == 1)
    warn = sum(1 for r in rows if r["scale_warn"] == 1)
    multi = sum(1 for r in rows if r["people_count"] and r["people_count"] > 1)
    confs = [r["mean_keypoint_conf"] for r in rows if r["mean_keypoint_conf"] != ""]
    tilts = [r["torso_tilt_deg"] for r in rows if r["torso_tilt_deg"] != ""]
    gap, longest = 0, 0                                  # 연속 누락(feature_ok=0) 최대 길이
    for r in rows:
        gap = gap + 1 if not r["feature_ok"] else 0
        longest = max(longest, gap)
    return {"frames": n, "pose_ok_ratio": pose / n if n else 0, "feature_ok_ratio": feat / n if n else 0,
            "body_ok_ratio": body / n if n else 0,
            "core_ok_ratio": core / n if n else 0, "scale_warn_ratio": warn / n if n else 0,
            "multi_person_frames": multi, "mean_conf": float(np.mean(confs)) if confs else float("nan"),
            "longest_gap": longest,
            "tilt_min": min(tilts) if tilts else "", "tilt_max": max(tilts) if tilts else ""}


def main() -> None:
    args = parse_args()
    with MANIFEST.open(encoding="utf-8-sig") as f:
        manifest = {r["clip_id"]: r for r in csv.DictReader(f)}
    if args.all:
        clips = list(manifest.values())
    else:
        missing = [c for c in args.clip if c not in manifest]
        if missing or not args.clip:
            sys.exit(f"--clip 또는 --all 을 주세요. Manifest 에 없는 clip_id: {missing}")
        clips = [manifest[c] for c in args.clip]

    cfg = load_config()
    kp_conf = args.kp_conf if args.kp_conf is not None else cfg["keypoint_confidence"]
    device = resolve_device(cfg["device"])
    model = YOLO(str(ROOT / cfg["pose_model"]))

    for i, clip in enumerate(clips, 1):
        out, rows = build(clip, model, cfg, device, kp_conf, args.max_frames, args.save_video)
        s = summarize(rows)
        print(f"[{i}/{len(clips)}] {clip['clip_id']} ({clip['action_label']}, {'정면' if clip['view'] == 'F' else '측면'})")
        print(f"  Frame {s['frames']} · Pose 성공 {s['pose_ok_ratio']:.1%} · Feature 성공 {s['feature_ok_ratio']:.1%}"
              f" · 어깨·골반 신뢰 {s['core_ok_ratio']:.1%} · 평균 Conf {s['mean_conf']:.3f}")
        print(f"  몸 크기 정규화 성공 {s['body_ok_ratio']:.1%} · 어깨 폭 경고 {s['scale_warn_ratio']:.1%} · 여러 사람 Frame {s['multi_person_frames']}"
              f" · 최장 연속 실패 {s['longest_gap']} Frame · 상체 기울기 {s['tilt_min']}~{s['tilt_max']}°")
        print(f"  Saved: {out.relative_to(ROOT)}  ({len(COLUMNS)}열)")


if __name__ == "__main__":
    main()
