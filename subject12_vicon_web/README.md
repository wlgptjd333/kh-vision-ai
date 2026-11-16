# 교과12 VICON → 웹 연동 (1·2일차)

비솔 스튜디오에서 촬영한 우리 조 동작 17개(`261111_D###_T###_LHY`, 동작별 마지막 테이크)를
Unreal에서 GLB로 내보낸 뒤 웹에서 재생하고, 웹캠으로 따라 한 동작과 비교합니다.

```
VICON FBX → Unreal (ViconActorFingers_Skeleton) → GLB 17개
   ↓ 1일차
FastAPI + Three.js 재생 → 30fps 3D 관절 JSON (output/json/)
   ↓ 2일차
Chrome 웹캠 → FastAPI /api/pose (YOLO11n-pose) → 관절각도 8개
   → VICON 기준 각도와 DTW 비교 → 점수·피드백 (output/results/)
```

## 폴더 구조

```
subject12_vicon_web/
├── app/
│   ├── main.py        # 1일차: 화면·GLB 제공, JSON 저장 API
│   ├── pose_api.py    # 2일차: /api/pose, /api/reference-angles, /api/compare
│   └── angles.py      # 2일차: Bone↔COCO 매핑, 관절각도, DTW, 점수 규칙
├── static/
│   ├── viewer.js      # 공통: GLB 로드·재생·3D 좌표 추출
│   ├── index.html     # 1일차 화면  (http://localhost:8000/)
│   └── compare.html   # 2일차 화면  (http://localhost:8000/compare)
├── scripts/
│   ├── check_day2.py       # JSON·각도·비교 API 점검
│   └── calibrate_score.py  # 동작 17개끼리 비교해 점수 기준 정하기
├── assets/            # GLB 17개 (Git 제외)
├── output/json/       # 1일차 결과 (Git 제외)
├── output/results/    # 2일차 결과 (Git 제외)
└── yolo11n-pose.pt    # 교과10에서 복사 (Git 제외)
```

## 데이터 (Git에 없음)

- `assets/*.glb`: 비솔 촬영 FBX(`LHY:` 네임스페이스 제거) → Unreal `ViconActorFingers_Skeleton`에 애니메이션 임포트 → GLB로 내보낸 17개 (약 100MB). 촬영 원본이라 올리지 않습니다.
- `yolo11n-pose.pt`: 교과10 폴더에서 복사 (`subject10_pose_action/yolo11n-pose.pt`).
- `output/`: 실행하면 생깁니다.

## 실행

```bash
cd ~/ai_vision/subject12_vicon_web
uv sync                       # 처음 한 번 (.venv 생성, Python 3.12)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- 서버 확인: http://localhost:8000/api/health → `glb_count: 17`

## 1일차 — GLB 재생 + 3D JSON 추출

1. Windows Chrome에서 http://localhost:8000 을 엽니다.
2. 동작을 고르면 재생됩니다. 마우스로 돌리거나 확대할 수 있습니다.
3. **3D JSON 추출**: 지금 동작을 서버(`output/json/`)에 저장하고 다운로드도 합니다.
4. **전체 추출**: 17개를 차례로 서버에 저장합니다. 오른쪽 목록이 `JSON 17/17`이 되면 끝입니다.

JSON 형식은 노트 3번과 같습니다: `fps 30`, `bone_count 73`, `frames[].joints[] = {name, x, y, z}` (Three.js 월드 좌표, Y-up, 미터).

## 2일차 — 웹캠 따라하기 비교

1. http://localhost:8000/compare 를 엽니다. 1일차 JSON이 있는 동작만 목록에 나옵니다.
2. **카메라 켜기** → Chrome 카메라 권한을 허용합니다. 전신이 보이도록 2~3m 떨어집니다.
3. 아래 표에서 **기준**(VICON) 각도와 **나**(웹캠) 각도가 실시간으로 비교됩니다.
4. **따라하기 녹화** → 3초 뒤 기준 동작이 처음부터 재생됩니다. 끝날 때까지 따라 합니다.
5. 점수, 관절별 평균 차이, 피드백이 나오고 결과가 `output/results/`에 저장됩니다.

- 웹캠은 Windows Chrome이 켜고 사진만 서버로 보냅니다. WSL에 웹캠을 연결할 필요가 없습니다.
- **좌우 반전 비교**: 기준 캐릭터를 거울처럼 마주 보고 따라 했다면 켭니다 (왼쪽↔오른쪽 각도를 바꿔서 비교).

### 비교 방식 (규칙 기반, 노트 4번 「방법 A」)

| 단계 | 내용 |
|---|---|
| 관절 맞추기 | VICON Bone → COCO 17: `LeftArm`=왼어깨, `LeftForeArm`=왼팔꿈치, `LeftHand`=왼손목, `LeftUpLeg`=왼엉덩이, `LeftLeg`=왼무릎, `LeftFoot`=왼발목 (오른쪽 동일) |
| 각도 8개 | 팔꿈치(어깨-팔꿈치-손목), 어깨(엉덩이-어깨-팔꿈치), 고관절(어깨-엉덩이-무릎), 무릎(엉덩이-무릎-발목) × 좌우 |
| 시간 맞추기 | 둘 다 10Hz로 다시 샘플링 → **DTW**로 빠르고 느린 차이를 맞춤 |
| 점수 | 맞춰진 프레임끼리 평균 각도 차이: 0° = 100점, 30° 이상 = 0점. OK < 8°, 보통 < 15°, 그 이상 NG |

점수 기준은 `scripts/calibrate_score.py`로 17개 기준 동작끼리 비교한 결과(다른 동작끼리 평균 차이 최소 7.5°, 중앙 18.6°)를 보고 정했습니다.

### 한계 (발표 때 같이 말할 것)

- VICON은 **3D 각도**, 웹캠은 **2D 각도**입니다. 카메라 정면에서 가장 잘 맞고, 옆으로 서면 차이가 커집니다.
- D001↔D002처럼 비슷한 동작은 평균 차이가 8° 정도라 이 점수만으로는 구분이 어렵습니다. 동작 종류를 자동으로 맞히려면 여러 사람 데이터로 Random Forest 등을 따로 학습해야 합니다 (노트 4번 7장).
- 손가락 Bone(73개 중 40개)은 YOLO Pose에 없어서 비교에 쓰지 않습니다.

## 핸드폰 영상 Pose confidence (11/13)

핸드폰으로 동작 17개를 앞·옆에서 찍은 영상(35개, 9번은 정면 재촬영 1개 추가)을 YOLO11n-pose(COCO 17)에 넣어
사람 박스 conf와 관절 conf를 정리합니다. 교과10 2일차 `10_pose_video_quality.py`와 같은 흐름입니다.

```bash
uv run python scripts/video_confidence.py "/mnt/c/Users/user1/Downloads/혜성" [--save-video] [--stride 2]
```

| 결과 (`output/video_conf/`) | 내용 |
|---|---|
| `summary.csv` | 영상마다 동작 번호·앞/옆, 검출률, 박스 conf, 관절 conf 평균, conf 0.5 이상 비율, 가장 나빴던 시각·관절 |
| `keypoints.csv` | 영상 × 관절 17개 평균 conf |
| `frames/<영상>.csv` | 프레임마다 `mean_keypoint_conf`, `low_keypoint_count`, `lowest_keypoint` + 관절 17개 conf |
| `samples/`, `videos/` | 뼈대와 conf를 그린 사진·영상 (초록 ≥0.8, 노랑 ≥0.5, 빨강 <0.5) |
| `keypoint_conf.png` | 앞/옆 관절별 평균 conf 그래프 |

- 앞/옆은 파일 이름(촬영 시각)으로 정합니다: 2분 안에 붙은 두 영상 = 한 동작(먼저 앞, 다음 옆), 짝 없는 영상 = 바로 앞 동작의 정면 재촬영.
- 결과(혜성 35개): 정면 평균 0.93, 옆 0.84. 옆모습은 카메라 반대쪽 귀(0.22)·눈(0.38)·손목(0.75)이 낮고, 바닥에 눕는 동작은 정면도 0.8대.

## 핸드폰 영상 Pose 품질 (11/13 촬영)

17개 동작을 핸드폰으로 앞·옆에서 찍은 영상 35개(9번 동작은 앞 재촬영 1개 추가)를 COCO 17 Keypoint로 돌려 confidence를 정리합니다. 교과10 2일차 `frame_quality.csv`와 같은 열을 씁니다.

```bash
uv run python scripts/video_confidence.py "/mnt/c/Users/user1/Downloads/혜성" --save-video
```

- `output/video_conf/summary.csv`: 동작 번호(촬영 시각으로 앞·옆 짝), 검출률, 박스 conf, 관절 conf 평균, 0.5 이상 비율, 가장 나쁜 시점
- `output/video_conf/frames/<영상>.csv`: 프레임마다 `mean_keypoint_conf`, `low_keypoint_count`, `lowest_keypoint` + 관절 17개 conf
- `output/video_conf/videos/`: 뼈대·conf를 그린 영상 (초록 ≥0.8, 노랑 0.5~0.8, 빨강 <0.5)
- 결과: 앞 평균 0.93, 옆 0.84. 옆모습은 카메라 반대쪽 귀·눈·손목이 낮음.

### YOLO vs MediaPipe (body-hero 설정)

```bash
uv run --with "mediapipe>=0.10.30,<0.11" python scripts/compare_mediapipe.py 영상폴더 \
    --lite pose_landmarker.task --full pose_landmarker_full.task
```

| 모델 | 검출률 | 점수 평균 | 떨림(작을수록 안정) | 좌우 뒤바뀜 |
|---|---|---|---|---|
| YOLO11n-pose | 99.9% | 0.884 | 0.0155 | 0.7% |
| MediaPipe Lite (480px) | 99.7% | 0.894 | 0.0051 | 0.15% |
| MediaPipe Full (480px) | 99.7% | 0.898 | 0.0048 | 0.19% |

- 점수는 YOLO conf와 MediaPipe visibility라 그대로 비교하면 안 됨 → 모델과 무관한 떨림(2차 차분 / 사람 키)도 같이 봄.
- MediaPipe는 mediapipe를 프로젝트 의존성에 넣지 않고 `uv run --with`로 임시로만 씀.
- WSL에 `libGLESv2.so.2`가 없으면: `apt download libgles2 && dpkg -x libgles2_*.deb ~/.local/mp_gles` 후 `LD_LIBRARY_PATH=~/.local/mp_gles/usr/lib/x86_64-linux-gnu`.

## 점검

```bash
uv run python scripts/check_day2.py [사람 사진.jpg]   # 서버 켠 상태에서
```

같은 동작을 넣으면 100점, 다른 동작(D001 vs D017)을 넣으면 35점 정도가 나오면 정상입니다.
