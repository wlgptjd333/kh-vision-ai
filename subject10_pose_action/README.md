# Subject 10 - Pose Estimation & Action Analysis

교과 10 자세추정 이론·실습 프로젝트입니다.

## 실행 준비

```bash
cd ~/ai_vision/subject10_pose_action
uv sync
source .venv/bin/activate
```

## 데이터 (Git에 올리지 않음)

강사님 저장소 [handgonpo/Vision_AI_Class](https://github.com/handgonpo/Vision_AI_Class)의 `교과10 자세추정/10일 수업/` 아래 `N일차 데이터/`에서 받아 넣습니다.

| 위치 | 넣을 파일 |
|---|---|
| `data/day01/` | `coco8-pose.zip` (압축 풀기: `python -m zipfile -e data/day01/coco8-pose.zip data/day01/coco8_pose`) |
| `data/day01/video/` | `solution_ci_pose_demo.mp4` |
| `data/day02/images/` | `front.jpg` `arms_up.jpg` `side.jpg` `partial_body.jpg` `occlusion.jpg` `distance_far.jpg` |
| `data/day02/videos/` | `pose_quality_demo.mp4` (= day01 영상 복사본) |

`yolo11n-pose.pt`는 첫 실행 때 Ultralytics가 자동으로 받습니다.

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

## Day 03 - Joint Angle, Normalization & Frame Feature

### 오늘 배운 내용

- 세 Keypoint로 관절각도 계산 (팔꿈치·무릎 좌우 4개)
- Hip Center 기준 몸 중심 이동 + Shoulder Width 기준 크기 정규화
- Near / Far, Person A / Person B 정규화 전·후 비교
- COCO Left / Right 기준과 Keypoint 좌우 반전
- 34개 Normalized 좌표 + 4개 관절각도 = **38-D Frame Feature**
- 동작을 설명하는 Feature 선택 (Arms Up: 손목 y − 어깨 y)
- 영상의 각 Frame에서 38-D Feature 생성 → 시간에 따른 변화 확인

### Day 03 핵심 흐름

```text
YOLO Pose → COCO 17 Keypoint + Confidence → 관절각도
→ Hip Center → Shoulder Width Normalization
→ Normalized 34 + Angle 4 = 38-D Frame Feature
→ 영상의 각 Frame에 반복 → Sequence의 재료
```

### 오늘 추가한 주요 코드

```text
src/feature_utils.py
scripts/11_check_day03_data.py ~ 18_video_frame_features.py
```

### 실행 결과 (11/16)

| 항목 | 결과 |
|---|---|
| 12 관절각도 (arms_up_person_a) | 팔꿈치 L/R 132.50° / 125.38°, 무릎 174.53° / 173.54° |
| 12 관절각도 (sit_pose) | 무릎 L/R 97.88° / 117.49° |
| 14 Near vs Far | 픽셀 평균거리 131.05 → 정규화 0.1435 |
| 14 Person A vs B | 69.40 → 0.3385 |
| 15 좌우 반전 | Left vs Right 0.6086 → 반전 후 0.1207 |
| 16 Frame Feature | 38개, 평균 Confidence 0.957 |
| 17 Arms Up | person_a True / **sit_pose도 True (오탐)** |
| 18 영상 Feature | 60 Frame 중 60 Valid, 왼팔꿈치 140.14° → 115.62° |

- 교재의 `18_video_frame_features.py` 코드 칸에는 17번 코드가 잘못 들어가 있어서
  교재 설명·결과에 맞춰 다시 작성함 (CSV: frame_id, time_sec, valid_pose, confidence 정보 + 38-D).
- Arms Up 규칙(손목이 어깨보다 위)은 앉은 자세에서도 True가 나옴 →
  Feature 한두 개로 만든 규칙은 쉽게 오탐하므로 Sequence·학습 모델이 필요한 이유.

### 오늘 이해한 핵심

Keypoint 좌표를 그대로 쓰지 않고 몸의 위치·크기 차이를 줄인 Normalized 좌표와 관절각도로
한 Frame의 자세를 38개의 숫자로 표현했다. 동작 구분에는 38개를 전부 보는 것보다
그 동작을 설명하는 Feature를 고르는 것이 중요하다.

### 다음 수업

4일차에는 여러 Frame의 Feature를 시간 순서로 묶어 하나의 동작을 표현하는 **Sequence**로 확장한다.
