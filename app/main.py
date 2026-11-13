"""교과12 VICON 웹 연동 — FastAPI 서버

1일차: GLB 17개를 Three.js로 재생하고, 프레임별 3D 관절 좌표 JSON을 output/json/에 저장
2일차: app/pose_api.py — 웹캠 YOLO Pose와 VICON 기준 동작을 관절각도로 비교

실행:
    uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# -------------------------------------
# 1. 프로젝트 경로 설정
# -------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"          # Unreal에서 Export한 GLB
STATIC_DIR = BASE_DIR / "static"          # index.html, compare.html, viewer.js
JSON_DIR = BASE_DIR / "output" / "json"   # 1일차 결과: 3D 관절 JSON
RESULTS_DIR = BASE_DIR / "output" / "results"  # 2일차 결과: 비교 점수

for d in (ASSETS_DIR, JSON_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 파일 이름에는 영문·숫자·_·- 만 허용 (../ 같은 경로 공격 방지)
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def glb_path(name: str) -> Path:
    if not NAME_RE.match(name):
        raise HTTPException(400, "파일 이름 형식이 올바르지 않습니다.")
    path = ASSETS_DIR / f"{name}.glb"
    if not path.is_file():
        raise HTTPException(404, f"assets/{name}.glb 파일이 없습니다.")
    return path


def json_path(name: str) -> Path:
    glb_path(name)  # 이름 검사 + GLB 존재 확인
    return JSON_DIR / f"{name}_3d.json"


# -------------------------------------
# 2. FastAPI 생성
# -------------------------------------

app = FastAPI(title="VICON Three.js Viewer")

app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -------------------------------------
# 3. 화면 제공
# -------------------------------------

@app.get("/")
def index():
    """1일차: GLB 재생 + 3D JSON 추출 화면"""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/compare")
def compare_page():
    """2일차: 웹캠 따라하기 비교 화면"""
    return FileResponse(STATIC_DIR / "compare.html", media_type="text/html")


# -------------------------------------
# 4. 서버 상태 · 동작 목록
# -------------------------------------

@app.get("/api/health")
def health():
    glbs = sorted(ASSETS_DIR.glob("*.glb"))
    return {
        "status": "ok",
        "glb_count": len(glbs),
        "json_count": len(list(JSON_DIR.glob("*_3d.json"))),
    }


@app.get("/api/actions")
def actions():
    """assets/ 의 GLB 목록과 JSON 추출 여부"""
    items = []
    for glb in sorted(ASSETS_DIR.glob("*.glb")):
        name = glb.stem
        items.append({
            "name": name,
            "glb": f"/assets/{glb.name}",
            "size_mb": round(glb.stat().st_size / 1_000_000, 1),
            "has_json": (JSON_DIR / f"{name}_3d.json").is_file(),
        })
    return items


# -------------------------------------
# 5. 추출한 3D JSON 저장 / 읽기
# -------------------------------------

@app.post("/api/save-json")
async def save_json(request: Request):
    data = await request.json()

    if not isinstance(data, dict) or not isinstance(data.get("source_glb"), str):
        raise HTTPException(400, "source_glb 항목이 없습니다.")

    name = data["source_glb"].removesuffix(".glb")
    out = json_path(name)  # assets에 실제로 있는 GLB인지 확인

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise HTTPException(400, "저장할 프레임 데이터가 없습니다.")

    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "saved": True,
        "file": out.name,
        "path": f"output/json/{out.name}",
        "frame_count": len(frames),
        "bone_count": data.get("bone_count"),
    }


@app.get("/api/json/{name}")
def get_json(name: str):
    path = json_path(name)
    if not path.is_file():
        raise HTTPException(404, f"{name}의 3D JSON이 아직 없습니다. 1일차 화면에서 먼저 추출하세요.")
    return FileResponse(path, media_type="application/json", filename=path.name)


# -------------------------------------
# 6. 2일차 API 연결 (YOLO Pose · 비교)
# -------------------------------------

try:
    from app.pose_api import router as pose_router
    app.include_router(pose_router)
except ImportError as e:  # 1일차 패키지만 설치한 경우에도 서버는 뜨게
    print(f"[2일차 API 비활성화] {e} — `uv sync`로 ultralytics를 설치하세요.")
