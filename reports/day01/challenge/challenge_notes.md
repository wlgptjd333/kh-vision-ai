# Day01 Team Mini Challenge

## 우리가 선택한 촬영 조건 (촬영 전 예상 먼저 적기)

| 촬영 조건 | 실행 전 예상 |
|---|---|
| (예: 한쪽 팔 올리기) | |
| (예: 옆으로 서기) | |
| (예: 앉기) | |

## 조건별 비교표

| 입력 | 조건 | 사람 검출 | BBox | Keypoint | Skeleton | 팀의 판단 |
|---|---|---|---|---|---|---|
| COCO 이미지 1 | 기존 데이터 | | | | | |
| COCO 이미지 2 | 기존 데이터 | | | | | |
| IP Webcam 1 | | | | | | |
| IP Webcam 2 | | | | | | |
| IP Webcam 3 | | | | | | |

## 가장 잘 나온 결과

- 파일:
- 관찰:

## 가장 불안정한 결과

- 파일:
- 관찰:
- 예상 원인:

## 오늘 확인한 Pipeline

JPG / MP4 / IP Webcam
→ Frame
→ YOLO Pose
→ Person Bounding Box
→ COCO 17 Keypoint
→ Skeleton
