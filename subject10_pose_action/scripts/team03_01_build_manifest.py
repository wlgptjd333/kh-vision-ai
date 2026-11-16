from __future__ import annotations

# ==================================================
# team03_01_build_manifest.py
# 3팀 핸드폰 영상 → Action·Subject·Take·View 를 정해 Manifest 만들기 (4일차 STEP 1)
# ==================================================
# 입력: 팀 드라이브의 "교과10 동작 영상/<이름>/<...>/YYYYMMDD_HHMMSS.mp4"
#   - 한 사람이 동작마다 정면(F) → 측면(S) 순서로 찍힘. 17개 동작을 4명이 돌아가며 촬영.
#   - 파일 이름의 촬영 시각으로 몇 번째 동작인지 정함 (ACTION_STARTS).
#   - 가로 영상은 측면. 세로 영상은 같은 동작 안에서 가로 영상이 있으면 정면,
#     없으면 시간 순서로 정면 → 측면.
#   - 같은 사람·동작·방향을 다시 찍었으면 Take 번호를 T01, T02 ... 로 붙임.
# 출력:
#   configs/team03_action_classes.csv   17개 동작 정의
#   configs/team03_clip_manifest.csv    MP4 한 개당 한 행 (이름 없음, 가명 S01~)
#   data/team_projects/team03/source_map.csv   clip_id ↔ 원본 경로 (실명 포함, Git 제외)
#   --copy 를 주면 data/team_projects/team03/rgb/videos/<clip_id>.mp4 로 복사
#
# 실행:
#   python scripts/team03_01_build_manifest.py               # 목록만 만들기
#   python scripts/team03_01_build_manifest.py --copy        # 영상 복사까지

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
TEAM_DATA = ROOT / "data" / "team_projects" / "team03"
DEFAULT_SOURCE = "/mnt/d/3팀드라이브/교과10 동작 영상"

# (action_id, action_label, 한글 이름, 분류) — 교과12 VICON 촬영 동작 17개 문서와 같은 순서
ACTIONS = [
    ("A01", "kiosk_slam", "키오스크 내려치기", "property_damage"),
    ("A02", "one_arm_throw", "한 팔로 물건 던지기", "property_damage"),
    ("A03", "freezer_front_kick", "냉동고 앞차기", "property_damage"),
    ("A04", "claw_machine_shake", "뽑기 기계 흔들기", "property_damage"),
    ("A05", "two_hand_shove", "양손으로 밀치기", "violence"),
    ("A06", "two_hand_weapon_swing", "도구 들고 양손 휘두르기", "violence"),
    ("A07", "sit_on_floor_mat", "바닥 매트에 주저앉기", "loitering"),
    ("A08", "jump_onto_chair_phone", "의자에 뛰어올라 앉아 휴대폰 보기", "loitering"),
    ("A09", "lie_prone_phone", "엎드려 휴대폰 보기", "loitering"),
    ("A10", "lie_supine_roll_side", "누워서 휴대폰 보다 옆으로 돌아눕기", "loitering"),
    ("A11", "body_check", "몸통 박치기", "violence"),
    ("A12", "kneel_grab_prize", "무릎 꿇고 뽑기 인형 꺼내기", "theft_suspect"),
    ("A13", "table_slam", "테이블 내려치기", "property_damage"),
    ("A14", "repeated_vomiting", "연속 구토하기", "emergency"),
    ("A15", "sleep_against_wall", "벽에 기대어 잠들기", "emergency"),
    ("A16", "faint_backward", "뒤로 기절하기", "emergency"),
    ("A17", "trip_fall_forward", "걸려 넘어지기 (앞으로)", "emergency"),
]

# 2026-11-13 촬영: 동작마다 4명이 차례로 찍은 구간의 시작 시각 (HHMMSS)
ACTION_STARTS = ["151200", "151640", "152300", "152700", "153150", "153600", "153900", "154500",
                 "160500", "161330", "161900", "162250", "162630", "164200", "164700", "165030",
                 "165800"]

# 폴더가 잘못된 파일 → 실제 촬영된 사람 (옷차림으로 확인)
SUBJECT_OVERRIDES = {
    "20261113_165054.mp4": ("S02", "uploaded in S03 folder; performer is S02 (checked by outfit)"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=DEFAULT_SOURCE, help="사람별 폴더가 있는 원본 영상 폴더")
    p.add_argument("--copy", action="store_true", help="T03 이름으로 data/ 아래에 복사")
    return p.parse_args()


def load_subjects(source: Path) -> dict[str, str]:
    """이름 → S01.. (실명은 data/ 아래 Git 제외 파일에만 둠). 처음이면 폴더 이름 순서로 만듦."""
    path = TEAM_DATA / "subjects.csv"
    if not path.exists():
        names = sorted(d.name for d in source.iterdir() if d.is_dir() and any(d.rglob("*.mp4")))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["subject_id", "name"])
            for i, n in enumerate(names, 1):
                w.writerow([f"S{i:02d}", n])
    with path.open(encoding="utf-8-sig") as f:
        return {r["name"]: r["subject_id"] for r in csv.DictReader(f)}


def action_of(stamp: str) -> str:
    hhmmss = stamp[9:15]
    idx = sum(1 for s in ACTION_STARTS if hhmmss >= s)
    if idx == 0:
        raise ValueError(f"촬영 시각이 첫 동작보다 이릅니다: {stamp}")
    return ACTIONS[idx - 1][0]


def is_landscape(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Frame 을 읽을 수 없습니다: {path}")
    h, w = frame.shape[:2]                     # OpenCV 가 회전 정보를 반영한 실제 화면 크기
    return w > h


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    subjects = load_subjects(source)

    clips = []
    for person_dir in sorted(d for d in source.iterdir() if d.is_dir()):
        for mp4 in sorted(person_dir.rglob("*.mp4")):
            sid, note = subjects[person_dir.name], ""
            if mp4.name in SUBJECT_OVERRIDES and SUBJECT_OVERRIDES[mp4.name][0] != sid:
                sid, note = SUBJECT_OVERRIDES[mp4.name]   # 아직 다른 사람 폴더에 있을 때만
            clips.append({"path": mp4, "stamp": mp4.stem, "subject_id": sid,
                          "action_id": action_of(mp4.stem), "landscape": is_landscape(mp4), "note": note})

    # 사람·동작별로 묶어서 정면/측면 정하기
    groups = defaultdict(list)
    for c in clips:
        groups[(c["subject_id"], c["action_id"])].append(c)
    for key, cs in groups.items():
        cs.sort(key=lambda c: c["stamp"])
        has_landscape = any(c["landscape"] for c in cs)
        for i, c in enumerate(cs):
            if c["landscape"]:
                c["view"] = "S"
            elif has_landscape or len(cs) == 1:
                c["view"] = "F"
            else:
                c["view"] = "F" if i % 2 == 0 else "S"
        takes = Counter()
        for c in cs:
            takes[c["view"]] += 1
            c["take_id"] = f"T{takes[c['view']]:02d}"
            if takes[c["view"]] > 1:
                c["note"] = (c["note"] + "; " if c["note"] else "") + "reshoot"

    labels = {a[0]: a[1] for a in ACTIONS}
    rows = []
    for c in sorted(clips, key=lambda c: (c["subject_id"], c["action_id"], c["view"], c["take_id"])):
        clip_id = f"T03_{c['subject_id']}_{c['action_id']}_{c['take_id']}_{c['view']}"
        rows.append({"clip_id": clip_id, "subject_id": c["subject_id"], "action_id": c["action_id"],
                     "action_label": labels[c["action_id"]], "take_id": c["take_id"], "view": c["view"],
                     "video_file": f"{clip_id}.mp4", "source_type": "RGB", "review_status": "PENDING",
                     "note": c["note"], "_src": c["path"], "_stamp": c["stamp"]})

    ids = [r["clip_id"] for r in rows]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        sys.exit(f"clip_id 중복: {dup}")

    # 빠진 정면/측면 표시 (사람 × 동작 × 방향)
    have = {(r["subject_id"], r["action_id"], r["view"]) for r in rows}
    missing = [(s, a[0], v) for s in sorted(set(subjects.values())) for a in ACTIONS for v in "FS"
               if (s, a[0], v) not in have]

    cfg = ROOT / "configs"
    with (cfg / "team03_action_classes.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["action_id", "action_label", "name_ko", "category"])
        w.writerows(ACTIONS)
    public = ["clip_id", "subject_id", "action_id", "action_label", "take_id", "view",
              "video_file", "source_type", "review_status", "note"]
    with (cfg / "team03_clip_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=public, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with (TEAM_DATA / "source_map.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "recorded_at", "source_path"])
        for r in rows:
            w.writerow([r["clip_id"], r["_stamp"], str(r["_src"])])

    if args.copy:
        out = TEAM_DATA / "rgb" / "videos"
        out.mkdir(parents=True, exist_ok=True)
        # 원본이 지워지거나 Take 번호가 바뀌면 예전 이름의 복사본은 지움 (원본은 건드리지 않음)
        keep = {r["video_file"] for r in rows}
        stale = [p for p in out.glob("*.mp4") if p.name not in keep]
        for p in stale:
            p.unlink()
        if stale:
            print(f"예전 이름 복사본 {len(stale)}개 정리")
        for i, r in enumerate(rows, 1):
            dst = out / r["video_file"]
            src_stat = r["_src"].stat()
            if (not dst.exists() or dst.stat().st_size != src_stat.st_size
                    or int(dst.stat().st_mtime) != int(src_stat.st_mtime)):
                shutil.copy2(r["_src"], dst)
            print(f"\r복사 {i}/{len(rows)}", end="", flush=True)
        print()

    print(f"MP4: {len(rows)}개 / 사람 {len(set(subjects.values()))}명 / 동작 {len(ACTIONS)}개")
    by_subject = Counter((r["subject_id"], r["view"]) for r in rows)
    for s in sorted(set(subjects.values())):
        print(f"  {s}: 정면 {by_subject[(s, 'F')]} · 측면 {by_subject[(s, 'S')]}")
    for r in rows:
        if r["note"]:
            print(f"  [참고] {r['clip_id']}: {r['note']}")
    for s, a, v in missing:
        print(f"  [없음] {s} {a} {'정면' if v == 'F' else '측면'}")
    print(f"Saved: {cfg / 'team03_clip_manifest.csv'}")


if __name__ == "__main__":
    main()
