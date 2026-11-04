# Day04 Mini Challenge

검사 이미지:
shift.jpg / rotate.jpg / distance.jpg / tilt_glare.jpg (실제 재촬영)

주요 변화:
위치 · 회전 · 거리(크기) · 기울기와 비닐 반사가 섞여 있음

선택한 방법:
SIFT + Homography (`src/32_day04_challenge.py`, `selected_method = "SIFT_HOMOGRAPHY"`)

선택 이유:
코드로 만든 Target 비교에서 Template Matching은 shift(0.4534)·rot15(0.4750)부터 엉뚱한 위치를 찾았고,
SIFT + Homography만 normal·shift·rot15·scale90 네 장 모두 로고 위치를 사각형으로 찾았다 (Inlier Ratio 0.97 이상).
실제 촬영은 여러 조건이 동시에 바뀌므로 회전·크기 변화에 강하고 위치까지 확인되는 방법을 골랐다.

결과:

| 이미지 | 주요 변화 | Good | Inlier | Ratio | 결과 | 관찰 |
|---|---|---:|---:|---:|---|---|
| `shift.jpg` | 위치 | 347 | 305 | 0.879 | FOUND | 로고를 감쌈 |
| `rotate.jpg` | 회전 | 389 | 360 | 0.925 | FOUND | 회전된 로고 방향대로 사각형이 기울어짐 |
| `distance.jpg` | 크기 | 365 | 343 | 0.940 | FOUND | 작게 찍힌 이미지(387×157)에서도 유지 |
| `tilt_glare.jpg` | 기울기·반사 | 294 | 212 | 0.721 | FOUND | Ratio가 가장 낮음, 사각형이 로고보다 약간 큼 |

관찰:
네 장 모두 FOUND. 다만 tilt_glare는 반사 부분에서 잘못된 Match가 늘어 Inlier Ratio가 0.72로 떨어졌다.
→ 비닐 반사·원근 변화가 가장 큰 위험 요인.

한계:
- Challenge 이미지가 로고 주변만 찍힌 근접 사진이라 실제 라인(제품 전체 + 배경)보다 쉬운 조건이다.
- 반사가 더 심하거나 로고 일부가 가려지면 Inlier가 더 줄 수 있다.
- 다른 제품(불량 로고)에서 Inlier가 얼마나 나오는지는 아직 측정하지 않았다.

Threshold:
Day05 Calibration 후 결정
