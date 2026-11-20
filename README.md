# vision-ai-lab

AI 과정 **후반부(프로젝트 과정, 2026-10-01~)** 실습과 과제를 모은 저장소입니다.
전반부 기초 과정(Python·데이터 분석·ML·DL 기초, 7/16~9/30)은 [ai-dev-basics](https://github.com/wlgptjd333/ai-dev-basics)에 있습니다.

## 교과별 폴더

| 교과 | 기간 | 폴더 | 내용 |
|---|---|---|---|
| 7·8 | 10/1~10/22 | [`subject07_08_labeling/`](subject07_08_labeling/README.md) | 조각김치 이물검출 팀 프로젝트 (라벨링 → YOLO). 저장소는 팀 저장소 링크 |
| 9 | 10/23~11/5 | [`subject09_machinevision/`](subject09_machinevision/README.md) | OpenCV 규칙 기반 머신비전, FastAPI AI 서빙, 평가과제 |
| 10 | 11/6~11/19 | [`subject10_pose_action/`](subject10_pose_action/README.md) | YOLO Pose 자세추정·동작분석 (1~7일차: Keypoint Dataset·QA·사람 단위 Split·Random Forest Baseline·Sliding Window 실시간 인식, uv 프로젝트). 3팀 팀 프로젝트는 [unmanned-store-anomaly](https://github.com/wlgptjd333/unmanned-store-anomaly) |
| 12 | 11/10~11/13 (비솔 촬영) | [`subject12_vicon_web/`](subject12_vicon_web/README.md) | VICON 모션캡처 GLB → FastAPI·Three.js 재생·3D JSON, 웹캠 YOLO Pose 따라하기 비교 |
| 백엔드 | 11/20~ | [`product_review_service/`](product_review_service/README.md) | 상품 리뷰 서비스: Django·DRF 서비스 서버 + FastAPI AI 서버 분리 (1단계 기본 구축) |
| 11·13~15 | 11/20~ | (예정) | 합성데이터, Edge AI, 제조검사 PBL, 산업안전 PBL |

## 따로 있던 저장소에서 합친 폴더

커밋 기록을 유지한 채 `git subtree`로 옮겼습니다 (2026-11-04, 교과10은 2026-11-06부터). 원래 저장소의 태그는 앞에 접두어를 붙여 옮겼습니다.

| 폴더 | 원래 저장소 | 태그 접두어 |
|---|---|---|
| `subject09_machinevision/rule_based/` | machine-vision-pipeline | `s09-rule/` |
| `subject09_machinevision/fastapi_sentiment/` | fastapi-sentiment-api | `s09-sentiment/` |
| `subject09_machinevision/fastapi_resnet18/` | (로컬 전용이었음) | `s09-resnet18/` |
| `subject10_pose_action/` | WSL `~/ai_vision/subject10_pose_action` (로컬, 수업 중 계속 커밋) | — |
| `subject12_vicon_web/` | WSL `~/ai_vision/subject12_vicon_web` (로컬) | — |
| `product_review_service/` | WSL `~/ai_vision/product-review-service` (로컬, 수업 중 계속 커밋) | — |

## 공통 규칙

- 하위 프로젝트마다 자기 `.venv`와 의존성 파일(`requirements.txt` 또는 교과10부터 `pyproject.toml`+`uv.lock`)을 둡니다. 실행은 해당 폴더에서 합니다.
- 데이터셋, 원본 사진, 모델 가중치, `.env`는 올리지 않습니다. 받는 방법은 각 폴더 README에 적습니다.
- `compose.yaml`의 DB 비밀번호는 내 PC Docker 안에서만 쓰는 로컬 실습용 기본값입니다.
