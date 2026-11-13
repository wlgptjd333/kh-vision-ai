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

## 점검

```bash
uv run python scripts/check_day2.py [사람 사진.jpg]   # 서버 켠 상태에서
```

같은 동작을 넣으면 100점, 다른 동작(D001 vs D017)을 넣으면 35점 정도가 나오면 정상입니다.
