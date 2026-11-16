from __future__ import annotations

# ==================================================
# team03_02_check_videos.py
# Manifest 의 모든 MP4 검사 (4일차 STEP 2)
# ==================================================
# - 파일 존재, 열림, FPS, 기록된 Frame 수, 실제로 끝까지 읽힌 Frame 수, 재생시간, 화면 크기
# - 실제 디코딩 Frame 수가 기록값과 3% (최소 10 Frame) 넘게 다르면 decode_ok = 0
# 입력: configs/team03_clip_manifest.csv + data/team_projects/team03/rgb/videos/
# 출력: reports/team_projects/team03/tables/video_inventory.csv
# 실행: python scripts/team03_02_check_videos.py

import csv
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "team03_clip_manifest.csv"
VIDEOS = ROOT / "data" / "team_projects" / "team03" / "rgb" / "videos"
OUT = ROOT / "reports" / "team_projects" / "team03" / "tables" / "video_inventory.csv"


def check(path: Path) -> dict:
    row = {"exists": int(path.exists()), "opened": 0, "fps": "", "frames_meta": "", "frames_read": "",
           "duration_sec": "", "width": "", "height": "", "orientation": "", "decode_ok": 0}
    if not path.exists():
        return row
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return row
    row["opened"] = 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()                     # 첫 Frame 으로 회전 반영된 실제 크기 확인
    n = 1 if ok else 0
    if ok:
        h, w = frame.shape[:2]
        row.update(width=w, height=h, orientation="portrait" if h > w else "landscape")
    while cap.grab():                          # 나머지는 디코딩만 (빠름)
        n += 1
    cap.release()
    row.update(fps=round(fps, 3), frames_meta=meta, frames_read=n,
               duration_sec=round(n / fps, 2) if fps else "",
               # 핸드폰 영상은 가변 FPS 라 기록된 Frame 수가 추정값 → 몇 Frame 차이는 정상
               decode_ok=int(n > 0 and abs(n - meta) <= max(10, meta * 0.03)))
    return row


def main() -> None:
    with MANIFEST.open(encoding="utf-8-sig") as f:
        clips = list(csv.DictReader(f))
    rows = []
    for i, c in enumerate(clips, 1):
        r = {k: c[k] for k in ("clip_id", "subject_id", "action_id", "action_label", "view", "video_file")}
        r.update(check(VIDEOS / c["video_file"]))
        rows.append(r)
        print(f"\r검사 {i}/{len(clips)}", end="", flush=True)
    print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if not r["decode_ok"]]
    durs = [r["duration_sec"] for r in rows if r["duration_sec"] != ""]
    print(f"MP4 {len(rows)}개 · 정상 {len(rows) - len(bad)} · 문제 {len(bad)}")
    print(f"재생시간 {min(durs):.1f}~{max(durs):.1f}초 · 평균 {sum(durs) / len(durs):.1f}초 · "
          f"Frame 합계 {sum(int(r['frames_read'] or 0) for r in rows):,}")
    print("화면:", dict(Counter((r["view"], r["orientation"]) for r in rows)))
    for r in bad:
        print(f"  [문제] {r['clip_id']} exists={r['exists']} opened={r['opened']} "
              f"meta={r['frames_meta']} read={r['frames_read']}")
    short = sorted(rows, key=lambda r: r["duration_sec"] or 0)[:3]
    print("짧은 영상:", [(r["clip_id"], r["duration_sec"]) for r in short])
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
