# Day09 검사기술 선택 Matrix

| 검사 문제 | 선택 기술 | 선택 이유 | 제외한 기술과 이유 |
|---|---|---|---|
| 부품 개수 | ROI + Threshold + Contour + Rule | 있는지와 개수는 면적·개수로 설명할 수 있다 | Anomaly Detection: 측정값으로 충분해 과하다 |
| 로고 정합 | SIFT + RANSAC + Homography | 회전·크기가 바뀌어도 위치까지 찾는다 | Template: 회전·크기 변화에 약하다 |
| SN / LOT | ROI + OCR + Regex + Expected | 읽은 값을 형식과 기준값으로 판정한다 | Anomaly Detection: 숫자가 틀린 것을 구분하지 못한다 |
| 미지의 표면 이상 | Pretrained Feature + Normal Reference | 불량을 미리 정의하지 않아도 된다 | Detection: 학습한 종류만 찾는다 |
| 위치가 계속 변하는 객체 | Object Detection (교과 8) | 위치와 종류를 함께 찾는다 | Template: 고정 위치를 전제로 한다 |
| Low-data 검사 | One-shot / Few-shot | 몇 장으로 시작할 수 있다 | 대규모 학습: 데이터가 부족하다 |
| 복합 검사 | 문제별 기술을 나눠 조합 | 문자·표면·위치는 실패 원인이 다르다 | 한 기술로 전부 처리: 설명과 개선이 어렵다 |

## Hybrid가 필요한 경우

두 방법이 서로 다른 샘플을 틀리고, FP와 FN 중 줄일 쪽이 정해져 있을 때

## 한 기술만 사용하는 것이 좋은 경우

한 측정값으로 충분히 판정되고 설명이 쉬울 때
