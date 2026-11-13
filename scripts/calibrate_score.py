"""점수 기준 정하기: 17개 기준 동작끼리 서로 비교했을 때 평균 각도 차이가 얼마나 나오는지
    uv run python scripts/calibrate_score.py
같은 동작 = 0°, 다른 동작끼리의 차이 분포를 보고 OK/보통/NG 경계를 정합니다.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import angles as A  # noqa: E402

refs = {}
for f in sorted((ROOT / "output" / "json").glob("*_3d.json")):
    refs[f.stem.split("_")[1]] = A.reference_angles(json.loads(f.read_text(encoding="utf-8")))

as_user = lambda ref: [{"t": t, "angles": {k: ref["angles"][k][i] for k in A.ANGLE_NAMES}} for i, t in enumerate(ref["times"])]
names = list(refs)
diffs = []
for a in names:
    row = []
    for b in names:
        d = A.compare(refs[a], as_user(refs[b]))["mean_diff_deg"] if a != b else 0.0
        row.append(d)
        if a != b:
            diffs.append(d)
    print(a, " ".join(f"{d:4.0f}" for d in row))

diffs = np.array(diffs)
print(f"\n다른 동작끼리 평균 각도 차이: 최소 {diffs.min():.1f}° / 하위10% {np.percentile(diffs, 10):.1f}° / 중앙 {np.median(diffs):.1f}° / 최대 {diffs.max():.1f}°")
