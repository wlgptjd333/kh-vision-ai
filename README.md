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
