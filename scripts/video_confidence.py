"""핸드폰 동영상 → YOLO Pose(COCO 17 keypoint) → confidence 정리.

    uv run python scripts/video_confidence.py "/mnt/c/Users/user1/Downloads/혜성"
    uv run python scripts/video_confidence.py 폴더 --stride 2 --save-video

결과 (output/video_conf/):
    summary.csv        동영상마다 쌍 번호·앞/옆(촬영 순서), 검출률, 사람 박스 conf, keypoint conf 평균, 앞/옆 추정
    keypoints.csv      동영상 × keypoint 17개 평균 conf
    frames/<영상>.csv  프레임마다 박스 conf + keypoint 17개 conf
    samples/<영상>.jpg 가운데 프레임에 뼈대·conf 그린 사진
    keypoint_conf.png  keypoint별 평균 conf 막대그래프 (앞/옆 나눠서)
    videos/<영상>.mp4  --save-video 일 때만
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from app.angles import COCO17  # noqa: E402

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
VIS_TH = 0.5  # 이 값 이상이면 "보인다"고 셈


def pick_largest(res):
    """사람이 여러 명이면 박스가 가장 큰 사람 하나."""
    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    i = int(area.argmax())
    kxy = res.keypoints.xy.cpu().numpy()[i]
    kcf = res.keypoints.conf
    kcf = kcf.cpu().numpy()[i] if kcf is not None else np.zeros(17)
    return {"box": xyxy[i], "box_conf": float(res.boxes.conf.cpu().numpy()[i]),
            "xy": kxy, "conf": kcf, "n_person": len(xyxy)}


def view_ratio(p):
    """어깨 너비 / 몸통 길이. 정면이면 크고(≈0.6~), 옆모습이면 작다(≈0.3 이하)."""
    xy, cf = p["xy"], p["conf"]
    if min(cf[5], cf[6], cf[11], cf[12]) < VIS_TH:
        return np.nan
    sh = np.linalg.norm(xy[5] - xy[6])
    torso = np.linalg.norm((xy[5] + xy[6]) / 2 - (xy[11] + xy[12]) / 2)
    return sh / torso if torso > 1 else np.nan


def draw(frame, p):
    img = frame.copy()
    s = max(img.shape[:2]) / 1000
    if p is None:
        cv2.putText(img, "no person", (20, int(60 * s)), 0, 1.5 * s, (0, 0, 255), max(2, int(3 * s)))
        return img
    xy, cf = p["xy"], p["conf"]
    x1, y1, x2, y2 = p["box"].astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), max(2, int(2 * s)))
    cv2.putText(img, f"person {p['box_conf']:.2f}", (x1, max(30, y1 - 10)), 0, 0.9 * s,
                (255, 200, 0), max(2, int(2 * s)))
    for a, b in SKELETON:
        if cf[a] >= VIS_TH and cf[b] >= VIS_TH:
            cv2.line(img, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), (0, 255, 0), max(2, int(3 * s)))
    for k in range(17):
        c = float(cf[k])
        col = (0, 200, 0) if c >= 0.8 else (0, 200, 255) if c >= VIS_TH else (0, 0, 255)
        pt = tuple(xy[k].astype(int))
        cv2.circle(img, pt, max(4, int(6 * s)), col, -1)
        cv2.putText(img, f"{c:.2f}", (pt[0] + 8, pt[1] - 4), 0, 0.5 * s, col, max(1, int(1.5 * s)))
    return img


def process(path, model, out, stride, save_video, device):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    rows, mid_img, mid_idx = [], None, n_total // 2
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % stride:
            continue
        res = model.predict(frame, verbose=False, device=device)[0]
        p = pick_largest(res)
        rows.append({"frame": i, "t": i / fps, "p": p})
        if mid_img is None and i >= mid_idx:
            mid_img = draw(frame, p)
        if save_video:
            if writer is None:
                (out / "videos").mkdir(exist_ok=True)
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out / "videos" / f"{path.stem}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))
            img = draw(frame, p)
            s = max(img.shape[:2]) / 1000
            cv2.putText(img, f"frame {i}  {i / fps:.2f}s", (20, img.shape[0] - int(30 * s)), 0,
                        1.0 * s, (255, 255, 255), max(2, int(2.5 * s)))
            writer.write(img)
    cap.release()
    if writer:
        writer.release()

    # 프레임별 CSV (교과10 2일차 frame_quality.csv 열 + 관절 17개 conf)
    worst = None  # (low_count, mean) 가 가장 나쁜 프레임
    with open(out / "frames" / f"{path.stem}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "time_sec", "people", "box_conf", "mean_keypoint_conf",
                    "low_keypoint_count", "lowest_keypoint", "lowest_keypoint_conf"] + COCO17)
        for r in rows:
            p = r["p"]
            if p is None:
                w.writerow([r["frame"], f"{r['t']:.3f}", 0, "", 0, 17, "NO_PERSON", 0] + [""] * 17)
                continue
            c = p["conf"]
            low, mean, j = int((c < VIS_TH).sum()), float(c.mean()), int(c.argmin())
            if worst is None or (low, -mean) > (worst[0], -worst[1]):
                worst = (low, mean, r["t"], COCO17[j])
            w.writerow([r["frame"], f"{r['t']:.3f}", p["n_person"], f"{p['box_conf']:.4f}",
                        f"{mean:.4f}", low, COCO17[j], f"{c[j]:.4f}"] + [f"{x:.4f}" for x in c])
    if mid_img is not None:
        cv2.imwrite(str(out / "samples" / f"{path.stem}.jpg"), mid_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    det = [r["p"] for r in rows if r["p"] is not None]
    kconf = np.array([p["conf"] for p in det]) if det else np.full((1, 17), np.nan)
    ratios = np.array([view_ratio(p) for p in det]) if det else np.array([np.nan])
    vr = float(np.nanmedian(ratios)) if np.isfinite(ratios).any() else np.nan
    view = "?" if np.isnan(vr) else "앞" if vr >= 0.45 else "옆"  # 누운 동작에선 틀림 → main에서 촬영 순서로 view 정함
    h, w_ = (mid_img.shape[:2] if mid_img is not None else (0, 0))
    return {
        "video": path.stem, "size": f"{w_}x{h}", "sec": round(n_total / fps, 1),
        "frames_used": len(rows), "detected_ratio": round(len(det) / max(1, len(rows)), 3),
        "multi_person_ratio": round(np.mean([p["n_person"] > 1 for p in det]), 3) if det else 0,
        "box_conf_mean": round(float(np.mean([p["box_conf"] for p in det])), 3) if det else 0,
        "kpt_conf_mean": round(float(np.nanmean(kconf)), 3),
        "kpt_conf_min_joint": COCO17[int(np.nanargmin(kconf.mean(0)))] if det else "",
        "visible_ratio": round(float(np.nanmean(kconf >= VIS_TH)), 3),
        "worst_time_sec": round(worst[2], 2) if worst else "",
        "worst_low_count": worst[0] if worst else "",
        "worst_lowest_keypoint": worst[3] if worst else "",
        "shoulder_torso_ratio": round(vr, 2) if not np.isnan(vr) else "",
        "view_guess_ratio": view,
        "_kpt_mean": kconf.mean(0),
    }


def stamp_sec(stem):
    """20261113_151338 → 초. 핸드폰 파일 이름이 촬영 시각."""
    try:
        t = stem.split("_")[1]
        return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
    except (IndexError, ValueError):
        return None


def assign_pairs(summ, gap):
    """촬영 시각이 gap초 안에 붙은 두 영상 = 같은 동작 (먼저 찍은 것 앞, 다음 옆)."""
    i, k = 0, 0
    while i < len(summ):
        k += 1
        a = stamp_sec(summ[i]["video"])
        b = stamp_sec(summ[i + 1]["video"]) if i + 1 < len(summ) else None
        if a is not None and b is not None and 0 <= b - a <= gap:
            summ[i]["pair"], summ[i]["view"] = k, "앞"
            summ[i + 1]["pair"], summ[i + 1]["view"] = k, "옆"
            i += 2
        elif k > 1:
            # 짝 없는 영상 = 바로 앞 동작을 정면에서 다시 찍은 것 (예: 161005 = 9번 재촬영)
            k -= 1
            summ[i]["pair"], summ[i]["view"] = k, "앞(재촬영)"
            i += 1
        else:
            summ[i]["pair"], summ[i]["view"] = k, "앞(단독)"
            i += 1


def plot(summ, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {g: np.array([s["_kpt_mean"] for s in summ if s["view"].startswith(g)]) for g in ("앞", "옆")}
    x = np.arange(17)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for k, (g, lab, off) in enumerate([("앞", "front", -0.2), ("옆", "side", 0.2)]):
        if len(groups[g]):
            ax.bar(x + off, np.nanmean(groups[g], 0), 0.4, label=f"{lab} (n={len(groups[g])})")
    ax.axhline(VIS_TH, color="gray", ls="--", lw=1)
    ax.set_xticks(x, COCO17, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean keypoint confidence")
    ax.set_title("YOLO11n-pose keypoint confidence (phone videos)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "keypoint_conf.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--stride", type=int, default=1, help="몇 프레임마다 하나씩 (1=전부)")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--pair-gap", type=int, default=120, help="앞·옆 한 쌍으로 볼 촬영 간격(초)")
    ap.add_argument("--model", default=str(BASE_DIR / "yolo11n-pose.pt"))
    ap.add_argument("--out", default=str(BASE_DIR / "output" / "video_conf"))
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(a.model)
    out = Path(a.out)
    for sub in ("frames", "samples"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    vids = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in (".mp4", ".mov", ".avi"))
    summ = []
    for k, v in enumerate(vids, 1):
        s = process(v, model, out, a.stride, a.save_video, device)
        summ.append(s)
        print(f"[{k}/{len(vids)}] {s['video']} det={s['detected_ratio']:.2f} "
              f"box={s['box_conf_mean']:.2f} kpt={s['kpt_conf_mean']:.2f} vis={s['visible_ratio']:.2f}",
              flush=True)

    assign_pairs(summ, a.pair_gap)
    cols = ["pair", "view"] + [c for c in summ[0] if not c.startswith("_") and c not in ("pair", "view")]
    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summ)
    with open(out / "keypoints.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["pair", "view", "video"] + COCO17)
        for s in summ:
            w.writerow([s["pair"], s["view"], s["video"]] + [f"{c:.3f}" for c in s["_kpt_mean"]])
    plot(summ, out)

    allk = np.array([s["_kpt_mean"] for s in summ])
    print("\n전체 keypoint 평균 conf")
    for name, c in sorted(zip(COCO17, np.nanmean(allk, 0)), key=lambda t: t[1]):
        print(f"  {name:15s} {c:.3f}")
    for g in ("앞", "옆"):
        ss = [s for s in summ if s["view"].startswith(g)]
        if ss:
            print(f"{g}: {len(ss)}개  kpt={np.mean([s['kpt_conf_mean'] for s in ss]):.3f}  "
                  f"vis={np.mean([s['visible_ratio'] for s in ss]):.3f}  "
                  f"box={np.mean([s['box_conf_mean'] for s in ss]):.3f}")
    print("\n쌍(동작)별 kpt conf")
    for k in sorted({s["pair"] for s in summ}):
        ss = [s for s in summ if s["pair"] == k]
        print(f"  {k:2d}: " + "  ".join(f"{x['view']} {x['video'][9:]} {x['kpt_conf_mean']:.2f}" for x in ss))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
