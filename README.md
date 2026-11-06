# Subject 10 - Pose Estimation & Action Analysis

교과 10 자세추정 이론·실습 프로젝트입니다.

## 실행 준비

```bash
cd ~/ai_vision/subject10_pose_action
uv sync
source .venv/bin/activate
```

## Day 01 - Pose 전체 구조·개발환경·실제 데이터 입력

### 사용한 입력

- COCO8-Pose
- Ultralytics Pose Demo MP4
- 스마트폰 IP Webcam 캡처 이미지

### 학습한 내용

- Pose Estimation과 Action Recognition의 차이
- WSL 프로젝트와 Python 가상환경 구성
- OpenCV / PyTorch / Ultralytics 확인
- GPU / CUDA 확인
- COCO8-Pose 이미지 입력
- YOLO Pose 추론
- Person Bounding Box
- COCO 17 Keypoint
- Skeleton
- 실제 MP4 Frame 처리
- MP4 Frame별 Pose
- IP Webcam 연결 및 Frame 저장
- 직접 촬영 이미지 Pose 추론

### Day 01 Pipeline

```text
JPG / MP4 / IP Webcam
        ↓
Frame
        ↓
YOLO Pose
        ↓
Person Bounding Box
        ↓
COCO 17 Keypoint
        ↓
Skeleton
```

### 주요 실행 파일

```text
scripts/
├── 00_check_environment.py
├── 01_image_input.py
├── 02_video_input.py
├── 03_pose_image.py
├── 04_pose_video.py
└── 05_ip_webcam_capture.py
```

### 주요 결과

- `reports/day01/environment.txt`
- COCO8-Pose 이미지 Pose 결과
- `reports/day01/pose_demo_output.mp4`
- IP Webcam 촬영 이미지 Pose 결과

### 다음 수업

2일차에는 Keypoint 좌표와 Confidence를 자세히 읽고,
촬영 조건에 따라 Pose 품질이 어떻게 달라지는지 분석합니다.

## Day 02 - Keypoint 좌표와 Pose 품질 분석

### 오늘 사용한 입력

- 강사가 준비한 조건별 이미지 6장
  - `front.jpg`
  - `arms_up.jpg`
  - `side.jpg`
  - `partial_body.jpg`
  - `occlusion.jpg`
  - `distance_far.jpg`
- 1일차에서 사용한 실제 MP4
- Team Mini Challenge 직접 촬영 이미지

### 오늘 학습한 내용

- Person BBox Confidence와 Keypoint Confidence 구분
- COCO 17 Keypoint Pixel 좌표 확인
- Normalized 좌표 이해
- Keypoint Confidence와 Threshold 이해
- Confidence 기준에 따른 Skeleton 표시
- 여러 이미지의 Pose 품질 CSV 비교
- BBox Area Ratio 비교
- 평균 Keypoint Confidence와 LOW 관절 확인
- MP4의 Frame별 Pose 품질 분석
- 불안정한 Frame을 숫자로 찾고 실제 영상과 연결
- 촬영 조건 하나를 바꾸어 Pose 품질 비교

### 실제 실습에서 확인한 결과

`09_compare_pose_images.py` 실행 결과 (yolo11n-pose, RTX 3060):

| 이미지 | BBox 비율 | 평균 Keypoint Conf | LOW 관절 |
|---|---:|---:|---:|
| front.jpg | 0.300 | 0.975 | 0 |
| arms_up.jpg | 0.675 | 0.960 | 0 |
| distance_far.jpg | 0.149 | 0.971 | 0 |
| occlusion.jpg | 0.265 | 0.953 | 1 |
| partial_body.jpg | 0.339 | 0.907 | 2 |
| side.jpg | 0.199 | 0.754 | 4 |

- `side.jpg`가 6장 중 가장 불안정 (좌·우 관절이 겹침)
- `front.jpg` VS `distance_far.jpg`: 화면상 크기는 크게 줄었지만 평균 Confidence는 `0.975 → 0.971`로 유지
- MP4 Frame 분석: 60개 Frame 모두 사람 검출, 약 0.80초(25~26번 Frame)에서 얼굴 주변 Keypoint Confidence가 낮아짐

### Day 02 Pipeline

```text
이미지 · MP4
        ↓
YOLO Pose
        ↓
Person BBox
+
COCO 17 Keypoint
        ↓
Pixel / Normalized 좌표
        ↓
Keypoint Confidence
        ↓
Threshold
        ↓
조건별 Pose 품질 비교
        ↓
Frame별 Pose 품질 분석
        ↓
숫자와 실제 이미지·영상 연결
```

### 주요 실행 파일

```text
scripts/
├── 06_check_day02_data.py
├── 07_pose_image_detail.py
├── 08_draw_skeleton_quality.py
├── 09_compare_pose_images.py
└── 10_pose_video_quality.py

src/
└── pose_utils.py
```

### 주요 결과 파일

```text
reports/day02/tables/
├── front_keypoints.csv
├── image_quality_summary.csv
├── pose_quality_demo_frame_quality.csv
└── pose_quality_demo_keypoints.csv

reports/day02/challenge/
└── challenge_notes.md
```

### 다음 수업

3일차에는 2일차에서 확인한 Keypoint를 이용하여
관절각도와 정규화된 Frame Feature를 만듭니다.
