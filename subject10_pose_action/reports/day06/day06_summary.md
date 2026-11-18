# Day 06 Action Recognition Summary

실행일 2026-11-18. 교재 `6일차_Action_Recognition_Baseline_성능평가`의 33~40번을 5일차 18개 Approved Dataset 으로 실행했습니다.
Mini Challenge(41번)는 [challenge/mini_challenge_result.md](challenge/mini_challenge_result.md).

## 1. Dataset

- Train Subject: S01 / Validation Subject: S02 / Test Subject: S03
- Train·Validation·Test Clips: 6 / 6 / 6
- Action Classes: 6 / Total Clips: 18
- 입력 점검(33번): Keypoint CSV 모두 OK, Action Coverage OK, Clip Leakage 없음

## 2. Frame Feature

- Normalized Coordinate: 34 (Hip Center 원점, Shoulder Width 로 크기 정규화)
- Joint Angle: 4 (팔꿈치·무릎 좌우)
- Frame Feature Count: 38

## 3. Original Baseline

- Sequence Steps: 30 (Clip 전체를 시간 0~1 로 펴서 30개 지점으로 보간)
- Sequence Feature Count: 1140 (30 × 38)
- min_valid_frames: 10
- Random Forest: n_estimators 300, max_depth null, random_state 42, class_weight balanced

## 4. Baseline Validation (36·37번)

| | 이 PC | 교재 |
|---|---:|---:|
| Accuracy | **0.5000** | 0.6667 |
| Macro F1 | **0.4444** | 0.5556 |
| 맞힌 Clip | 3 | 4 |

오분류 (확신도는 6개 Class 중 가장 높은 확률):

| Clip | 정답 → 예측 | 확신도 | 교재에서도 틀림? |
|---|---|---:|---|
| S02_A003_T01 | walk_turn_walk → leg_raise_lower | 0.21 | 예 |
| S02_A004_T01 | drink_return → wave | 0.23 | 예 |
| S02_A006_T01 | wave → drink_return | 0.22 | **아니오 (이 PC 만)** |

- drink_return 과 wave 는 둘 다 "한 팔을 머리 높이로 올렸다 내리는" 동작이라 좌표가 비슷합니다.
  원본 영상에서도 S02 는 물병을 입에, 손을 머리 위로 — 팔꿈치 높이 차이 정도만 다릅니다.
- walk_turn_walk 는 걷는 동안 한쪽 다리가 앞으로 나가 leg_raise_lower 의 다리 들기와 비슷해 보입니다.
- 틀린 3개 모두 확신도가 0.21~0.23 으로 낮습니다 (Class 6개라 우연이면 0.17).

### 교재와 결과가 다른 이유

- Train 이 6개(동작당 1개)뿐이라 Random Forest 의 난수만 바꿔도 결과가 크게 흔들립니다.
  같은 Baseline 에서 `random_state` 만 0~19 로 바꿔 보니 Validation Accuracy 가
  **0.50 (1번) · 0.67 (7번) · 0.83 (9번) · 1.00 (3번)**, 평균 0.78 이었습니다 (`challenge/seed_sensitivity.csv`).
- 교재의 0.6667 과 이 PC 의 0.5000 은 모두 이 범위 안입니다. 이 PC 와 교재는 YOLO 버전·GPU 계산에 따라
  Keypoint 소수점이 조금 다르고(예: S02_A004 single_person_ratio 0.2987 vs 0.3052), 그 작은 차이가
  확신도 0.22 vs 0.20 처럼 아슬아슬한 Clip(wave)의 예측을 바꿉니다.

## 5. Static VS Sequence (38번)

| | Feature 수 | Accuracy | Macro F1 |
|---|---:|---:|---:|
| 가운데 Frame 1개 (Static) | 38 | 0.6667 | 0.5556 |
| 30-Step Sequence | 1140 | 0.5000 | 0.4444 |

- 교재는 둘 다 0.6667. 이 PC 에서는 Static 이 한 Clip 더 맞혔지만, 위 난수 변동(0.5~1.0) 안의 차이라
  "Static 이 낫다"고 결론 내리지 않습니다.
- 참고: 3팀 데이터(170개, 사람 단위 4-fold)에서는 Sequence 74% vs Static 57% 로 Sequence 가 분명히 높았습니다.
  데이터가 작으면 이런 차이가 보이지 않습니다.

## 6. Feature Set Comparison (39번)

| | Feature 수 | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Coordinates + Angles | 1140 | 0.5000 | 0.4444 |
| Coordinates Only | 1020 | **1.0000** | **1.0000** |

- 교재와 같은 방향 (교재 0.6667 → 1.0000).
- 하지만 Validation Clip 이 6개라 1개 = 16.7%p. 각도는 좌표로 계산되는 값이라 새 정보가 적고,
  Train 6개에서는 Feature 1,140 개 중 어떤 것이 뽑히느냐가 결과를 좌우합니다. 일반 결론으로 확대하지 않습니다.
  (3팀 170개에서도 각도 유무 차이는 난수 변동 안이었습니다.)

## 7. Mini Challenge

- 변경한 조건: Sequence Step
- 변경 전 → 변경 후: 30 → 15 (Feature 1140 → 570), 나머지 설정 동일
- 실험 전 가설: Train 이 6개뿐이라 Step 을 줄여도 결과는 비슷하거나 한두 Clip 만 바뀔 것
- Baseline: Accuracy 0.5000 / Macro F1 0.4444
- 변경 후: Accuracy 0.6667 / Macro F1 0.5833
- 예측이 바뀐 Clip: wave (drink_return → **wave 정답**), drink_return (wave → leg_raise_lower, 여전히 오답)
- 결과 해석: 한 Clip 차이(16.7%p)는 난수 변동보다 작으므로 "15 Step 이 낫다"고 보지 않음

## 8. Final Selected Config

- Sequence Steps: 30
- Sequence Feature Count: 1140
- min_valid_frames: 10
- n_estimators: 300
- max_depth: null
- random_state: 42

최종 조건을 선택한 이유:

- Static, 각도 제외, 15 Step 모두 Baseline 과의 차이가 Random Forest 난수 변동(0.5~1.0) 안이라
  어느 쪽이 낫다고 판단할 근거가 없음 → 교재 Baseline 을 그대로 둠

## 9. Final Test (40번)

- Test Subject: S03
- Test Clips: 6
- Accuracy: **1.0000**
- Macro F1: **1.0000**
- Correct Clips: 6 / Misclassified Clips: 0 (확신도 0.23~0.37)

Test 결과를 본 뒤 조건을 다시 변경하지 않은 이유:

- Test 는 처음 보는 사람에 대한 마지막 확인용입니다. Test 를 보고 설정을 바꾸면 Test 가 Validation 이 되어
  성능을 과대평가하게 됩니다.
- 참고: 이 PC 에서는 40번을 Mini Challenge(41번)보다 먼저 실행했습니다. Mini Challenge 는 Validation 만 썼고
  최종 설정도 바꾸지 않았으므로 Test 가 선택에 쓰이지 않았습니다.
- Validation 0.50 vs Test 1.00 처럼 크게 달라도 오류가 아닙니다. 사람이 다르고 Clip 이 6개뿐이기 때문입니다.

## 10. 현재 Dataset의 한계

- 3 Subjects × 6 Actions × 1 Take 의 작은 교육용 Dataset 입니다.
- Action 당 Subject 별 Clip 이 1개뿐이라 1개 = 16.7%p 입니다.
- 현재 Accuracy 를 현업 성능으로 해석하지 않습니다.

## 11. 6일차에서 배운 핵심

- 길이가 다른 Clip 을 30 Step 으로 보간해 같은 길이(30 × 38 = 1,140)의 입력으로 만든다.
- 평가는 Accuracy 와 Macro F1 을 같이 보고, 혼동 행렬과 오분류 Clip 을 원본 영상과 연결해 원인을 찾는다.
- 비교 실험은 한 번에 조건 하나만 바꾸고, 그 차이가 **우연한 변동(난수·Clip 1개)보다 큰지** 확인해야 한다.
  데이터가 작으면 Random Forest 의 난수 하나로도 50%~100% 가 나온다.

## 12. 다음 수업

7일차에서는 실시간 Frame 을 계속 모아 Sliding Window 를 만들고 실시간 Action Recognition 으로 연결합니다.
