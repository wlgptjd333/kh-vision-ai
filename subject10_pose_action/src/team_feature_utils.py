from __future__ import annotations

# ==================================================
# team_feature_utils.py — 3팀(무인 상점 이상행동) 추가 Feature
# ==================================================
# 기본 38-D (feature_utils.py) 는 그대로 두고, 3팀에 필요한 값만 여기에 더한다.
# 4일차 교재 3팀 항목: "38-D + 상체 기울기 + 원본 위치", "누운 자세·측면·발목 누락 QA"
#
#   torso_length      어깨 중심 ↔ 골반 중심 거리(px). 측면에서 어깨 폭 대신 쓸 수 있는 기준 길이
#   shoulder_ratio    어깨 폭 / 몸통 길이. 정면 ≈ 0.6~0.9, 측면 ≈ 0.1~0.3
#                     → 작으면 어깨 폭 정규화 값이 크게 튀므로 scale_warn 으로 표시
#   torso_tilt_deg    골반→어깨 방향과 화면 위쪽 사이 각도. 0 = 똑바로 섬, 90 = 누움, 180 = 거꾸로
#   body_aspect       Keypoint 들을 감싸는 상자의 높이 / 너비. 서 있으면 큼, 누우면 작음
#   hip_x_img/y_img   골반 중심의 화면 위치 (0~1). 정규화로 지워지는 "원본 위치" 보존

import math

import numpy as np

from src.feature_utils import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER, midpoint

TEAM03_FEATURE_NAMES = ["torso_length", "shoulder_ratio", "torso_tilt_deg", "body_aspect",
                        "hip_x_img", "hip_y_img"]
SCALE_WARN_RATIO = 0.35        # 어깨 폭이 몸통 길이의 35% 보다 작으면 어깨 폭 정규화를 믿기 어려움

# --------------------------------------------------
# 몸 크기(body scale) 정규화 — 3팀 영상용
# --------------------------------------------------
# 어깨 폭만 쓰면 측면에서, 몸통 길이만 쓰면 카메라 쪽으로 눕거나 숙일 때 기준 길이가 0 에 가까워져 값이 튄다.
# (11/17 표본 18,576 Frame: 골반→가장 먼 관절 거리 / 기준 길이 가 5 를 넘는 비율
#   어깨 폭 = 정면 4.1% · 측면 56.7%,  몸통 길이 = 정면 7.5% · 측면 0.5%,  아래 방식 = 둘 다 0.01%)
# → 두 길이를 같은 단위로 맞춘 뒤 더 큰 쪽을 쓴다.
#   똑바로 선 정면에서 어깨 폭 ≈ 몸통 길이 × 0.70 이므로 어깨 폭 / 0.70 을 몸통 길이로 환산한 값으로 본다.
BODY_SHOULDER_RATIO = 0.70
BODY_SCALE_MIN_PX = 5.0        # 기준 길이가 이보다 작으면 정규화하지 않음


def body_scale(keypoints) -> tuple[float, str]:
    """원본 Keypoint(px) → (기준 길이 px, 어느 쪽을 썼는지 'torso' | 'shoulder')"""
    kp = np.asarray(keypoints, dtype=np.float32)
    torso_len = float(np.linalg.norm(midpoint(kp[LEFT_HIP], kp[RIGHT_HIP])
                                     - midpoint(kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER])))
    from_shoulder = float(np.linalg.norm(kp[LEFT_SHOULDER] - kp[RIGHT_SHOULDER])) / BODY_SHOULDER_RATIO
    return (torso_len, "torso") if torso_len >= from_shoulder else (from_shoulder, "shoulder")


def normalize_by_body_scale(keypoints):
    """골반 중심을 원점으로 옮기고 body_scale 로 나눈 17×2 좌표, 기준 길이, 출처를 돌려준다."""
    kp = np.asarray(keypoints, dtype=np.float32)[:, :2]
    scale, source = body_scale(kp)
    if scale < BODY_SCALE_MIN_PX:
        raise ValueError("몸 크기 기준 길이를 계산할 수 없습니다.")
    center = midpoint(kp[LEFT_HIP], kp[RIGHT_HIP])
    return (kp - center) / scale, scale, source


def team03_features(keypoints, confidence, img_w: int, img_h: int, kp_conf: float = 0.5) -> dict:
    """원본 Keypoint(px) 17×2 → 3팀 추가 Feature dict. 계산할 수 없으면 빈 칸("")."""
    kp = np.asarray(keypoints, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    out = {name: "" for name in TEAM03_FEATURE_NAMES}

    shoulder_c = midpoint(kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER])
    hip_c = midpoint(kp[LEFT_HIP], kp[RIGHT_HIP])
    torso = hip_c - shoulder_c
    torso_len = float(np.linalg.norm(torso))
    sw = float(np.linalg.norm(kp[LEFT_SHOULDER] - kp[RIGHT_SHOULDER]))

    out["hip_x_img"] = round(float(hip_c[0]) / img_w, 4)
    out["hip_y_img"] = round(float(hip_c[1]) / img_h, 4)
    if torso_len > 1e-6:
        out["torso_length"] = round(torso_len, 2)
        out["shoulder_ratio"] = round(sw / torso_len, 4)
        up = shoulder_c - hip_c                      # 골반 → 어깨 (화면 y 는 아래로 커짐)
        out["torso_tilt_deg"] = round(math.degrees(math.atan2(abs(up[0]), -up[1])), 2)

    seen = kp[conf >= kp_conf]                       # 믿을 만한 관절만으로 상자
    if len(seen) >= 5:
        w = float(seen[:, 0].max() - seen[:, 0].min())
        h = float(seen[:, 1].max() - seen[:, 1].min())
        if w > 1e-6:
            out["body_aspect"] = round(h / w, 4)
    return out
