# Day 06 Mini Challenge

재현: `python scripts/41_day06_mini_challenge.py` (34번 코드를 그대로 불러 Step 만 15 로, Test 는 쓰지 않음)

## 1. 사용한 데이터

- Train Subject: S01
- Validation Subject: S02
- Train Clip 수: 6
- Validation Clip 수: 6
- Action 수: 6

## 2. Sequence Dataset

- Sequence Steps: 30 (Baseline) / 15 (실험)
- Frame Feature 수: 38
- Sequence Feature 수: 1140 / 570
- Train CSV 행 수: 6
- Validation CSV 행 수: 6

## 3. Random Forest 학습

- Train Clips: 6
- Feature Count: 1140
- Classes: bend_return, drink_return, leg_raise_lower, sit_stand, walk_turn_walk, wave
- 생성된 Model: `artifacts/day06/rf_action_baseline.joblib` (Git 제외)

## 4. Baseline Validation

- Accuracy: 0.5000
- Macro F1: 0.4444
- 정답 Clip 수: 3
- 오답 Clip 수: 3

## 5. 오분류

| 실제 Action | 예측 Action | Confidence | 원본 MP4 에서 확인한 내용 |
|---|---|---:|---|
| walk_turn_walk | leg_raise_lower | 0.21 | 걷는 중 한쪽 다리가 앞으로 나간 Frame 이 많아 다리 들기와 비슷 |
| drink_return | wave | 0.23 | 물병을 든 팔을 입 높이로 올림 — 손을 머리 위로 드는 wave 와 팔 모양이 비슷 |
| wave | drink_return | 0.22 | 반대 방향으로 같은 혼동 (교재에서는 맞힌 Clip) |

## 6. 본수업 Feature 비교 결과

- Static VS Sequence: Static 0.6667 / Sequence 0.5000 (교재는 둘 다 0.6667)
- Coordinates + Angles VS Coordinates Only: 0.5000 / 1.0000 (교재 0.6667 / 1.0000)
- 결과를 일반화하면 안 되는 이유: Validation 이 6개라 1개 = 16.7%p 이고,
  Random Forest 의 `random_state` 만 0~19 로 바꿔도 Baseline Validation 이 0.50~1.00 (평균 0.78)로 흔들림
  → 실험 사이 차이가 이 변동보다 작음

## 7. 한 변수 실험

- 바꾼 조건: Sequence Step
- 변경 전: 30 (Feature 1140)
- 변경 후: 15 (Feature 570)
- 가설: Train 이 6개뿐이라 Step 을 줄여도 결과는 비슷하거나 한두 Clip 만 바뀔 것

## 8. 실험 결과

| 항목 | Baseline | Mini Experiment |
|---|---|---|
| Sequence Steps | 30 | 15 |
| Feature 수 | 1140 | 570 |
| n_estimators | 300 | 300 |
| max_depth | null | null |
| Validation Accuracy | 0.5000 | 0.6667 |
| Validation Macro F1 | 0.4444 | 0.5833 |
| 오분류 Clip 수 | 3 | 2 |

- 예측이 바뀐 Clip: wave (drink_return → wave, 정답으로), drink_return (wave → leg_raise_lower, 여전히 오답)
- Baseline 보다: 한 Clip 좋아짐 — 하지만 난수 변동 범위 안
- 이유에 대한 생각: 6개 동작은 3~10초짜리 단순 동작이라 15단계로도 모양이 충분히 남습니다.
  Feature 가 절반이 되면서 Random Forest 가 고르는 Feature 조합이 달라졌고, 확신도 0.22 근처에서
  아슬아슬하던 wave 가 정답 쪽으로 넘어간 것으로 봅니다. 성능이 좋아졌다고 판단하지 않고 Baseline(30)을 유지합니다.

## 9. Test를 사용하지 않은 이유

- 조건을 고르는 데 Test(S03)를 쓰면 Test 가 사실상 Validation 이 되어 처음 보는 사람에 대한 성능을 과대평가하게 됩니다.
  Challenge 는 S02 Validation 만 사용했습니다.

## 10. 오늘 다시 이해한 핵심

- 한 번에 하나만 바꾸고, 그 차이가 우연한 변동보다 큰지 먼저 확인한다.
- 데이터가 작을 때는 "좋아졌다/나빠졌다"보다 "판단할 수 없다"가 정직한 결론일 때가 많다.
