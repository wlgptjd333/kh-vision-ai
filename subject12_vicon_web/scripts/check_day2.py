"""2일차 점검: 서버를 켠 상태에서 실행
    uv run python scripts/check_day2.py [사람 사진 경로]

1) 1일차 JSON이 시간에 따라 실제로 변하는지 (모든 프레임이 같으면 추출 실패)
2) 기준 각도가 그럴듯한지 (0~180°)
3) 기준 동작을 그대로 '나'로 넣으면 100점 근처, 다른 동작을 넣으면 점수가 내려가는지
4) (사진을 주면) /api/pose가 관절 17개와 각도를 돌려주는지
"""

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import angles as A  # noqa: E402

API = "http://localhost:8000"


def post(path, body: bytes, ctype="application/json"):
    req = urllib.request.Request(API + path, data=body, headers={"Content-Type": ctype})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


files = sorted((ROOT / "output" / "json").glob("*_3d.json"))
print(f"JSON {len(files)}개")
refs = {}
for f in files:
    data = json.loads(f.read_text(encoding="utf-8"))
    hips = [next(j for j in fr["joints"] if j["name"] == "Hips") for fr in data["frames"]]
    moved = max(abs(h["x"] - hips[0]["x"]) + abs(h["z"] - hips[0]["z"]) for h in hips)
    ref = A.reference_angles(data)
    vals = [v for k in A.ANGLE_NAMES for v in ref["angles"][k] if v is not None]
    refs[f.stem.removesuffix("_3d")] = ref
    print(f"  {f.stem:32s} {data['frame_count']:4d}프레임 bone {data['bone_count']}  "
          f"Hips 이동 {moved:.2f}m  각도 {min(vals):.0f}~{max(vals):.0f}°")

names = list(refs)
a, b = refs[names[0]], refs[names[-1]]
as_user = lambda ref: [{"t": t, "angles": {k: ref["angles"][k][i] for k in A.ANGLE_NAMES}} for i, t in enumerate(ref["times"])]

same = A.compare(a, as_user(a))
diff = A.compare(a, as_user(b))
print(f"\n같은 동작끼리: {same['score']}점 (평균 차이 {same['mean_diff_deg']}°)")
print(f"{names[0]} vs {names[-1]}: {diff['score']}점 (평균 차이 {diff['mean_diff_deg']}°) → {diff['feedback']}")

api = post("/api/compare", json.dumps({"action": names[0], "frames": as_user(a)[::3]}).encode())
print(f"/api/compare (같은 동작, 10fps): {api['score']}점, 저장 {api['saved']}")

if len(sys.argv) > 1:
    r = post("/api/pose", Path(sys.argv[1]).read_bytes(), "image/jpeg")
    print(f"\n/api/pose: found={r['found']} 관절 {len(r['keypoints'])}개 {r['infer_ms']}ms")
    print("  각도:", r["angles"])
