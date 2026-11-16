"""같은 핸드폰 영상에서 YOLO11n-pose vs MediaPipe Pose Landmarker(body-hero 설정) 비교.

    uv run --with "mediapipe>=0.10.30,<0.11" python scripts/compare_mediapipe.py 영상폴더 \
        --lite ~/body-hero/tools/pose_landmarker.task --full ~/body-hero/tools/pose_landmarker_full.task

- MediaPipe는 body-hero(tools/udp_send_webcam_ml.py)처럼 긴 변 480으로 줄여서 VIDEO 모드로 돌림.
- 점수가 서로 다른 값이라(YOLO=keypoint conf, MediaPipe=visibility) 숫자를 바로 비교하면 안 됨.
  그래서 모델과 상관없는 지표도 같이 봄:
    det_ratio   사람을 찾은 프레임 비율
    score_mean  COCO 17개 위치의 점수 평균 (YOLO conf / MP visibility)
    jitter      관절 떨림: 2차 차분 |x(t+1) - 2x(t) + x(t-1)| / 사람 키, 중앙값 (작을수록 안정)
    swap_ratio  왼/오 뒤바뀜 의심 프레임 비율 (어깨·엉덩이 좌우 순서가 한 프레임 만에 뒤집힘)
결과: output/mp_compare/summary.csv, samples/<영상>.jpg (같은 프레임 좌: YOLO, 우: MediaPipe)
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


def yolo_runner(model, device):
    def run(frame):
        r = model.predict(frame, verbose=False, device=device)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return None
        b = r.boxes.xyxy.cpu().numpy()
        i = int(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax())
        return r.keypoints.xy.cpu().numpy()[i], r.keypoints.conf.cpu().numpy()[i]
    return run


def mp_runner(task_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(task_path)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.55, min_tracking_confidence=0.7,
        min_pose_presence_confidence=0.5)  # body-hero 값
    lm = vision.PoseLandmarker.create_from_options(opts)

    def run(frame, ts_ms):
        h, w = frame.shape[:2]
        s = 480 / max(h, w)  # body-hero: 480x360으로 처리
        small = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        res = lm.detect_for_video(img, ts_ms)
        if not res.pose_landmarks:
            return None
        pts = res.pose_landmarks[0]
        xy = np.array([[pts[k].x * w, pts[k].y * h] for k in MP_TO_COCO])
        vis = np.array([pts[k].visibility for k in MP_TO_COCO])
        return xy, vis
    run.close = lm.close
    return run


def metrics(seq, n):
    det = [s for s in seq if s is not None]
    out = {"det_ratio": len(det) / max(1, n)}
    if not det:
        return {**out, "score_mean": np.nan, "score_ge_0.5": np.nan, "jitter": np.nan, "swap_ratio": np.nan}
    sc = np.array([d[1] for d in det])
    out["score_mean"] = float(sc.mean())
    out["score_ge_0.5"] = float((sc >= 0.5).mean())
    jit, swaps, pairs = [], 0, 0
    for a, b, c in zip(seq, seq[1:], seq[2:]):
        if a is None or b is None or c is None:
            continue
        H = np.ptp(b[0][:, 1]) or 1.0  # 사람 키(픽셀) 대신 관절 y범위
        ok = (a[1] >= .5) & (b[1] >= .5) & (c[1] >= .5)
        if ok.any():
            acc = np.linalg.norm(c[0] - 2 * b[0] + a[0], axis=1)[ok] / H
            jit.append(np.median(acc))
    for a, b in zip(seq, seq[1:]):
        if a is None or b is None:
            continue
        pairs += 1
        for l, r in ((5, 6), (11, 12)):
            if np.sign(a[0][l, 0] - a[0][r, 0]) != np.sign(b[0][l, 0] - b[0][r, 0]) and \
               abs(b[0][l, 0] - b[0][r, 0]) > 0.05 * (np.ptp(b[0][:, 1]) or 1):
                swaps += 1
                break
    out["jitter"] = float(np.median(jit)) if jit else np.nan
    out["swap_ratio"] = swaps / max(1, pairs)
    return out


def draw(img, d, title):
    img = img.copy()
    s = max(img.shape[:2]) / 1000
    cv2.putText(img, title, (20, int(60 * s)), 0, 1.4 * s, (255, 255, 255), max(2, int(3 * s)))
    if d is None:
        cv2.putText(img, "no person", (20, int(120 * s)), 0, 1.4 * s, (0, 0, 255), max(2, int(3 * s)))
        return img
    xy, cf = d
    for a, b in SKELETON:
        if cf[a] >= .5 and cf[b] >= .5:
            cv2.line(img, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), (0, 255, 0), max(2, int(3 * s)))
    for k in range(17):
        col = (0, 200, 0) if cf[k] >= .8 else (0, 200, 255) if cf[k] >= .5 else (0, 0, 255)
        cv2.circle(img, tuple(xy[k].astype(int)), max(4, int(7 * s)), col, -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--lite", required=True)
    ap.add_argument("--full")
    ap.add_argument("--stride", type=int, default=2, help="60fps 영상을 30fps처럼 (body-hero 웹캠 30FPS)")
    ap.add_argument("--out", default=str(BASE_DIR / "output" / "mp_compare"))
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    yolo = yolo_runner(YOLO(str(BASE_DIR / "yolo11n-pose.pt")), device)
    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    models = [("mp_lite", a.lite)] + ([("mp_full", a.full)] if a.full else [])

    rows = []
    vids = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() == ".mp4")
    for k, v in enumerate(vids, 1):
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        mps = {name: mp_runner(path) for name, path in models}
        seqs = {"yolo": [], **{n: [] for n in mps}}
        mid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
        sample, i = None, -1
        while True:
            ok, f = cap.read()
            if not ok:
                break
            i += 1
            if i % a.stride:
                continue
            seqs["yolo"].append(yolo(f))
            for n, run in mps.items():
                seqs[n].append(run(f, int(i * 1000 / fps)))
            if sample is None and i >= mid:
                sample = np.hstack([draw(f, seqs["yolo"][-1], "YOLO11n-pose")] +
                                   [draw(f, seqs[n][-1], f"MediaPipe {n[3:]}") for n in mps])
        cap.release()
        for run in mps.values():
            run.close()
        if sample is not None:
            sc = 1600 / sample.shape[1]
            cv2.imwrite(str(out / "samples" / f"{v.stem}.jpg"), cv2.resize(sample, None, fx=sc, fy=sc))
        n = len(seqs["yolo"])
        line = [f"[{k}/{len(vids)}] {v.stem}"]
        for name, seq in seqs.items():
            m = metrics(seq, n)
            rows.append({"video": v.stem, "model": name, **{x: round(y, 4) for x, y in m.items()}})
            line.append(f"{name}: det={m['det_ratio']:.2f} score={m['score_mean']:.2f} "
                        f"jit={m['jitter']:.4f} swap={m['swap_ratio']:.3f}")
        print("  ".join(line), flush=True)

    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n모델별 평균 (영상 35개)")
    for name in ["yolo"] + [m[0] for m in models]:
        rs = [r for r in rows if r["model"] == name]
        print(f"  {name:8s} " + "  ".join(f"{x}={np.nanmean([r[x] for r in rs]):.4f}"
                                          for x in ("det_ratio", "score_mean", "score_ge_0.5", "jitter", "swap_ratio")))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
