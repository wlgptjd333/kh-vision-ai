"""2일차 API: 웹캠 프레임 → YOLO Pose → 관절각도, VICON 기준 동작과 비교

브라우저(Windows Chrome)가 웹캠을 켜고 JPEG 프레임을 보내므로
WSL에 웹캠(/dev/video0)을 연결하지 않아도 됩니다.
"""

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app import angles as A
from app.main import BASE_DIR, RESULTS_DIR, json_path

router = APIRouter(prefix="/api")

MODEL_PATH = BASE_DIR / "yolo11n-pose.pt"
_model = None
_lock = threading.RLock()  # detect()가 잡은 상태에서 get_model()도 잡으므로 RLock


def get_model():
    """처음 요청 때 한 번만 YOLO Pose 모델을 불러옴"""
    global _model
    with _lock:
        if _model is None:
            from ultralytics import YOLO
            if not MODEL_PATH.is_file():
                raise HTTPException(500, f"{MODEL_PATH.name}이 없습니다. 프로젝트 폴더에 넣어 주세요.")
            _model = YOLO(str(MODEL_PATH))
        return _model


# -------------------------------------
# 1. 웹캠 한 장 → 사람 관절 17개 + 각도
# -------------------------------------

def detect(jpeg: bytes) -> dict:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "이미지를 읽을 수 없습니다.")
    h, w = img.shape[:2]

    with _lock:  # GPU 모델 하나를 여러 요청이 동시에 쓰지 않게
        result = get_model()(img, verbose=False, conf=0.4)[0]

    if result.keypoints is None or len(result.boxes) == 0:
        return {"found": False, "width": w, "height": h, "keypoints": [], "angles": {}}

    # 여러 명이면 가장 큰 사람 한 명만
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    i = int(np.argmax(areas))
    xy = result.keypoints.xy[i].cpu().numpy()
    conf = result.keypoints.conf[i].cpu().numpy()

    return {
        "found": True,
        "width": w,
        "height": h,
        "box": [round(float(v), 1) for v in boxes[i]],
        "keypoints": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for (x, y), c in zip(xy, conf)],
        "angles": A.angles_from_yolo(xy, conf),
    }


@router.post("/pose")
async def pose(request: Request):
    """body = JPEG 바이트 (Content-Type: image/jpeg)"""
    jpeg = await request.body()
    if not jpeg:
        raise HTTPException(400, "이미지가 비어 있습니다.")
    t0 = time.perf_counter()
    out = await run_in_threadpool(detect, jpeg)
    out["infer_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


# -------------------------------------
# 2. VICON 기준 동작 각도 (1일차 JSON 사용)
# -------------------------------------

_ref_cache: dict[str, tuple[float, dict]] = {}


def load_reference(name: str) -> dict:
    path = json_path(name)
    if not path.is_file():
        raise HTTPException(404, f"{name}의 3D JSON이 없습니다. 1일차 화면(/)에서 먼저 추출하세요.")
    mtime = path.stat().st_mtime
    cached = _ref_cache.get(name)
    if cached and cached[0] == mtime:
        return cached[1]
    ref = A.reference_angles(json.loads(path.read_text(encoding="utf-8")))
    _ref_cache[name] = (mtime, ref)
    return ref


@router.get("/reference-angles/{name}")
def reference_angles(name: str):
    return load_reference(name)


# -------------------------------------
# 3. 따라하기 녹화 결과 비교
# -------------------------------------

@router.post("/compare")
async def compare(request: Request):
    body = await request.json()
    name = body.get("action")
    frames = body.get("frames")
    if not isinstance(name, str) or not isinstance(frames, list):
        raise HTTPException(400, "action, frames 항목이 필요합니다.")

    frames = [f for f in frames if isinstance(f, dict) and isinstance(f.get("angles"), dict) and "t" in f]
    if len(frames) < 10:
        raise HTTPException(400, f"녹화된 프레임이 너무 적습니다 ({len(frames)}개). 카메라에 전신이 보이게 다시 해 보세요.")

    ref = load_reference(name)
    result = await run_in_threadpool(A.compare, ref, frames, bool(body.get("mirror")))
    result["action"] = name

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{name}_{stamp}.json"
    out.write_text(json.dumps({**result, "frames": frames}, ensure_ascii=False, indent=2), encoding="utf-8")
    result["saved"] = f"output/results/{out.name}"
    return result
