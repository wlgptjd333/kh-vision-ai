"""2일차: VICON 73 Bone ↔ YOLO Pose COCO 17 관절을 같은 '관절각도'로 바꿔서 비교

- VICON JSON(1일차 결과)은 3D 좌표, 웹캠 YOLO Pose는 2D 좌표입니다.
  그래서 좌표를 직접 비교하지 않고, 사람 크기·카메라 위치와 무관한 '관절각도'로 바꿉니다.
- 주의: 3D 각도와 2D 각도는 카메라를 정면으로 볼 때 가장 비슷합니다.
  옆으로 서면 2D 각도가 작게 나올 수 있습니다 (노트 4번: 카메라 각도 검증 필요).
"""

from __future__ import annotations

import warnings

import numpy as np

# YOLO Pose COCO 17 관절 순서
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# VICON Bone → COCO 관절 (Bone의 시작점 = 관절 위치)
VICON_TO_COCO = {
    "nose": "Head",
    "left_shoulder": "LeftArm", "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
    "left_hip": "LeftUpLeg", "right_hip": "RightUpLeg",
    "left_knee": "LeftLeg", "right_knee": "RightLeg",
    "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
}

# 각도 이름 → (끝점 A, 꼭짓점 B, 끝점 C): B에서 A와 C가 이루는 각도
ANGLES = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}
ANGLE_NAMES = list(ANGLES)

KOREAN = {
    "left_elbow": "왼쪽 팔꿈치", "right_elbow": "오른쪽 팔꿈치",
    "left_shoulder": "왼쪽 어깨", "right_shoulder": "오른쪽 어깨",
    "left_hip": "왼쪽 고관절", "right_hip": "오른쪽 고관절",
    "left_knee": "왼쪽 무릎", "right_knee": "오른쪽 무릎",
}


# -------------------------------------
# 1. 세 점으로 각도 계산 (2D·3D 공통)
# -------------------------------------

def joint_angle(a, b, c) -> float | None:
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    ba, bc = a - b, c - b
    n = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n < 1e-9:
        return None
    cos = np.clip(np.dot(ba, bc) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def angles_from_points(points: dict) -> dict:
    """points: {coco 관절 이름: 좌표 또는 None} → {각도 이름: 도 또는 None}"""
    out = {}
    for name, (a, b, c) in ANGLES.items():
        pa, pb, pc = points.get(a), points.get(b), points.get(c)
        out[name] = None if pa is None or pb is None or pc is None else round(joint_angle(pa, pb, pc), 1)
    return out


def mirror_angles(angles: dict) -> dict:
    """왼쪽 ↔ 오른쪽 바꾸기 (거울처럼 마주 보고 따라 할 때)"""
    swap = lambda k: k.replace("left_", "TMP_").replace("right_", "left_").replace("TMP_", "right_")
    return {swap(k): v for k, v in angles.items()}


# -------------------------------------
# 2. YOLO 키포인트 → 각도
# -------------------------------------

def angles_from_yolo(xy: np.ndarray, conf: np.ndarray, min_conf: float = 0.5) -> dict:
    points = {
        name: (xy[i] if conf[i] >= min_conf else None)
        for i, name in enumerate(COCO17)
    }
    return angles_from_points(points)


# -------------------------------------
# 3. VICON 3D JSON → 프레임별 각도
# -------------------------------------

def reference_angles(data: dict) -> dict:
    times, series = [], {k: [] for k in ANGLE_NAMES}
    for frame in data["frames"]:
        joints = {j["name"]: (j["x"], j["y"], j["z"]) for j in frame["joints"]}
        points = {coco: joints.get(bone) for coco, bone in VICON_TO_COCO.items()}
        ang = angles_from_points(points)
        times.append(frame["time_sec"])
        for k in ANGLE_NAMES:
            series[k].append(ang[k])
    return {
        "source_glb": data.get("source_glb"),
        "fps": data.get("fps"),
        "duration_sec": data.get("duration_sec"),
        "times": times,
        "angles": series,
    }


# -------------------------------------
# 4. 시간 맞추기: 같은 간격으로 다시 샘플링
# -------------------------------------

def to_matrix(times, angle_rows: list[dict], hz: float = 10.0) -> np.ndarray:
    """시간이 불규칙한 각도 기록 → (프레임 수, 8) 행렬, 빈 값은 앞뒤로 보간"""
    t = np.asarray(times, dtype=float)
    grid = np.arange(t.min(), t.max() + 1e-9, 1.0 / hz)
    mat = np.full((len(grid), len(ANGLE_NAMES)), np.nan)
    for j, k in enumerate(ANGLE_NAMES):
        v = np.array([np.nan if r.get(k) is None else r[k] for r in angle_rows], dtype=float)
        ok = ~np.isnan(v)
        if ok.sum() >= 2:
            mat[:, j] = np.interp(grid, t[ok], v[ok])
    return mat


# -------------------------------------
# 5. DTW: 속도가 달라도 같은 동작 구간끼리 맞춰서 비교
# -------------------------------------

def dtw_path(ref: np.ndarray, user: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 전부 빈 값인 칸의 nanmean 경고
        cost = np.nanmean(np.abs(ref[:, None, :] - user[None, :, :]), axis=2)
    cost = np.nan_to_num(cost, nan=90.0)

    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])

    # 끝에서부터 거꾸로 따라가며 맞춰진 프레임 쌍을 찾음
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return path[::-1]


# -------------------------------------
# 6. 최종 비교: 점수 + 관절별 차이 + 피드백
# -------------------------------------

def feedback_text(name: str, user_mean: float, ref_mean: float) -> str:
    part = KOREAN[name]
    less = user_mean < ref_mean
    if name.endswith(("elbow", "knee")):
        tip = "더 펴세요" if less else "더 굽히세요"
    elif name.endswith("shoulder"):
        tip = "팔을 더 올리세요" if less else "팔을 조금 내리세요"
    else:  # hip: 각도가 작을수록 상체를 많이 숙인 것
        tip = "상체를 더 세우세요" if less else "상체를 더 숙이세요"
    return f"{part}: {tip}"


def compare(ref: dict, user_frames: list[dict], mirror: bool = False, hz: float = 10.0) -> dict:
    ref_rows = [{k: ref["angles"][k][i] for k in ANGLE_NAMES} for i in range(len(ref["times"]))]
    ref_mat = to_matrix(ref["times"], ref_rows, hz)

    rows = [mirror_angles(f["angles"]) if mirror else f["angles"] for f in user_frames]
    user_mat = to_matrix([f["t"] for f in user_frames], rows, hz)

    detected = sum(1 for r in rows if any(v is not None for v in r.values()))

    path = dtw_path(ref_mat, user_mat)
    ri = np.array([p[0] for p in path])
    ui = np.array([p[1] for p in path])
    r, u = ref_mat[ri], user_mat[ui]

    per_angle = {}
    for j, k in enumerate(ANGLE_NAMES):
        ok = ~np.isnan(r[:, j]) & ~np.isnan(u[:, j])
        if ok.sum() == 0:
            per_angle[k] = None
            continue
        per_angle[k] = {
            "name_ko": KOREAN[k],
            "mean_diff": round(float(np.mean(np.abs(r[ok, j] - u[ok, j]))), 1),
            "ref_mean": round(float(np.mean(r[ok, j])), 1),
            "user_mean": round(float(np.mean(u[ok, j])), 1),
        }

    diffs = [v["mean_diff"] for v in per_angle.values() if v]
    mean_diff = float(np.mean(diffs)) if diffs else 90.0
    # 규칙 (scripts/calibrate_score.py 결과로 정함):
    #   기준 동작 17개를 서로 비교하면 다른 동작끼리도 평균 차이가 8~40°(중앙 19°)
    #   → 평균 차이 0° = 100점, 30° 이상 = 0점 (선형)
    score = round(max(0.0, 100.0 - mean_diff * 100.0 / 30.0), 1)

    worst = sorted((v["mean_diff"], k) for k, v in per_angle.items() if v)[::-1]
    feedback = [
        feedback_text(k, per_angle[k]["user_mean"], per_angle[k]["ref_mean"]) + f" (평균 {d:.0f}° 차이)"
        for d, k in worst[:3] if d >= 15
    ] or ["기준 동작과 비슷합니다. 잘했어요!"]

    return {
        "score": score,
        "mean_diff_deg": round(mean_diff, 1),
        "grade": "OK" if mean_diff < 8 else ("보통" if mean_diff < 15 else "NG"),
        "per_angle": per_angle,
        "feedback": feedback,
        "ref_duration_sec": round(float(ref["times"][-1]), 2),
        "user_duration_sec": round(float(user_frames[-1]["t"] - user_frames[0]["t"]), 2),
        "user_frames": len(user_frames),
        "detected_ratio": round(detected / len(user_frames), 2),
        "mirror": mirror,
    }
