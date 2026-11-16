from __future__ import annotations

# ==================================================
# 18_video_frame_features.py
# 영상의 각 Frame → 38-D Frame Feature → CSV (한 행 = 한 Frame)
# ==================================================
# 교재(3일차 26번)의 이 파일 코드 칸에는 17번(Arms Up 이미지 판정)과
# 같은 코드가 잘못 들어가 있어서, 교재 설명·실행 명령·결과 해석에 맞춰 다시 작성함.
#   - 실행:  python scripts/18_video_frame_features.py --max-frames 150
#   - 입력:  data/day03/videos/feature_demo.mp4 (2일차 pose_quality_demo.mp4 복사본)
#   - 출력:  reports/day03/tables/feature_demo_frame_features.csv
#   - 열:    frame_id, time_sec, valid_pose, mean_keypoint_conf, low_keypoint_count,
#            hip_center_x/y, shoulder_width + 38-D Feature (16번과 같은 이름·순서)
#   - Pose 를 못 찾거나 정규화가 안 되면 valid_pose = 0 으로 남기고 Feature 칸은 비움
#     (Confidence 가 조금 낮다고 Frame 을 버리지는 않고 개수만 기록)

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
from src.feature_utils import normalize_by_shoulder_width, selected_joint_angles  # noqa: E402

# 16번의 make_feature_names() 를 그대로 재사용 (파일 이름이 숫자로 시작해 import 대신 경로로 불러옴)
_spec = importlib.util.spec_from_file_location("frame_feature16", ROOT / "scripts" / "16_make_frame_feature.py")
_m16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m16)
make_feature_names = _m16.make_feature_names

ANGLE_NAMES = ["left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "data" / "day03" / "videos" / "feature_demo.mp4"))
    parser.add_argument("--max-frames", type=int, default=150, help="처리할 최대 Frame 수 (0 = 전체)")
    parser.add_argument("--kp-conf", type=float, default=0.50)
    return parser.parse_args()


def frame_feature(result, kp_conf):
    """YOLO 결과 하나 → (meta dict, 38개 값) / Pose·정규화 실패면 (meta, None)"""
    kps = result.keypoints
    if kps is None or kps.xy is None or len(kps.xy) == 0:
        return {"mean_keypoint_conf": "", "low_keypoint_count": ""}, None

    keypoints = kps.xy[0].cpu().numpy()                      # 첫 번째 사람 (16번과 같은 기준)
    confidence = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(17, dtype=np.float32)
    meta = {
        "mean_keypoint_conf": round(float(np.mean(confidence)), 4),
        "low_keypoint_count": int(np.sum(confidence < kp_conf)),
    }
    try:
        normalized, center, scale = normalize_by_shoulder_width(keypoints)
        angles = selected_joint_angles(keypoints)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return meta, None

    values = [float(v) for point in normalized for v in point[:2]] + [float(angles[n]) for n in ANGLE_NAMES]
    if len(values) != 38 or not np.all(np.isfinite(values)):
        return meta, None
    meta.update({
        "hip_center_x": round(float(center[0]), 4),
        "hip_center_y": round(float(center[1]), 4),
        "shoulder_width": round(float(scale), 4),
    })
    return meta, values


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    feature_names = make_feature_names()
    meta_names = ["frame_id", "time_sec", "valid_pose", "mean_keypoint_conf", "low_keypoint_count",
                  "hip_center_x", "hip_center_y", "shoulder_width"]
    rows, total, valid = [], 0, 0

    while True:
        if args.max_frames > 0 and total >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        result = model.predict(source=frame, imgsz=config["image_size"],
                               conf=config["person_confidence"], device=device, verbose=False)[0]
        meta, values = frame_feature(result, args.kp_conf)

        row = {name: "" for name in meta_names + feature_names}
        row.update(meta)
        row["frame_id"] = total
        row["time_sec"] = round((total - 1) / fps, 3)
        row["valid_pose"] = 1 if values is not None else 0
        if values is not None:
            valid += 1
            for name, value in zip(feature_names, values):
                row[name] = round(value, 6)
        rows.append(row)
    cap.release()

    output_path = ROOT / "reports" / "day03" / "tables" / f"{source.stem}_frame_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=meta_names + feature_names)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Source: {source.name}")
    print(f"Total frames: {total}")
    print(f"Valid pose frames: {valid}")
    print(f"Feature count per frame: {len(feature_names)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
