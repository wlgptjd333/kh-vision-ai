# Day 05 Summary

실행일 2026-11-18. 교재 `5일차_18개 Action Clip을 Keypoint Dataset으로 만들고 Subject-wise Split하기`의 25~32번을
강사님 제공 18개 MP4(볼트 `10일수업/5일차 데이터`)로 그대로 실행했습니다.

## 1. Dataset

- Subjects: S01, S02, S03
- Actions: 6 (A001 bend_return, A002 leg_raise_lower, A003 walk_turn_walk, A004 drink_return, A005 sit_stand, A006 wave)
- Take: Action당 1
- Total Clips: 18 (모두 640×480, 25fps, 62~268 Frame)
- S01 의 6개는 4일차 영상과 같은 파일 (파일 크기 동일)

## 2. Keypoint Dataset

- 생성된 Keypoint CSV: 18개 (`data/day05/keypoints/`, Git 제외)
- COCO Keypoint: 17개
- 저장 값: Pixel 좌표, Image Normalized 좌표, Confidence

## 3. 자동 QA (27번)

- AUTO_PASS: 13
- CHECK: 5 — 교재와 같은 Clip (S02 A001~A005)
- Pose Ratio: 18개 모두 1.0
- CHECK 원인: `single_person_ratio` (교재 값과 거의 같음)

| Clip | single_person_ratio (이 PC) | 교재 |
|---|---:|---:|
| S02_A001_T01 | 0.0000 | 0.0000 |
| S02_A002_T01 | 0.0000 | 0.0000 |
| S02_A003_T01 | 0.0581 | 0.0581 |
| S02_A004_T01 | 0.2987 | 0.3052 |
| S02_A005_T01 | 0.0083 | 0.0083 |

## 4. Human QA

- CHECK 5개: 영상 3시점에 YOLO 상자를 모두 그려 확인 → **오른쪽 옷걸이에 걸린 외투를 사람으로 잡은 것**
  (상자 confidence 0.26~0.72). 실제 촬영 대상은 1명이고 동작도 정상 → **PASS 5**
- AUTO_PASS 13개: 시작·중간·끝 장면 확인 → 전신이 보이고 동작을 끝까지 수행 → **PASS 13**
- RESHOOT 0 / EXCLUDE 0

> 교재 설명(23.9)은 AUTO_PASS 13개를 UNREVIEWED 로 두어도 된다고 하지만, 교재에 실린 28·29번 코드는
> `human_decision == PASS` 만 승인합니다(그대로 두면 Approved 5개 → 30번에서 S01·S03 없음 오류).
> 그래서 교재 23.7("18개뿐이므로 교육 실습에서는 전체를 확인")대로 18개 모두 사람이 확인해 PASS 로 기록했습니다.
> 판단 근거는 `data/day05/qa/keypoint_clip_review.csv` 의 `human_note` 에 있습니다.

## 5. Approved Dataset

- Approved Clips: 18개 (S01·S02·S03 각 6, Action 별 3)

## 6. Subject-wise Split

- Train: S01 / 6 Clips
- Validation: S02 / 6 Clips
- Test: S03 / 6 Clips
- 각 Split 에 A001~A006 모두 존재

## 7. Leakage 검사 (31번)

- Subject overlap: 없음
- Clip overlap: 없음
- Action Coverage: 정상
- Result: Leakage check: OK

## 8. Random Frame Split 비교 (32번)

- Train Frames: 1946 / Test Frames: 487 (교재와 같음)
- Subject overlap: S01, S02, S03
- Clip overlap: 18

Frame 을 무작위로 나누면 같은 사람·같은 영상의 Frame 이 Train 과 Test 에 함께 들어갑니다.
그래서 평가는 사람 기준으로 나눕니다.

## 9. 현재 Dataset의 한계

3 Subjects × 6 Actions × 1 Take = 18 Clips 의 최소 교육용 Dataset 입니다.
Pipeline 을 배우기에는 적절하지만 일반화 성능을 판단하기에는 작습니다.

## 10. Mini Challenge

교재의 5일차 Mini Challenge(팀 영상으로 같은 흐름)는 3팀 저장소에서 170개 영상으로 수행했습니다.
→ [unmanned-store-anomaly](https://github.com/wlgptjd333/unmanned-store-anomaly) (동작별 관절 QA, Human QA PASS 170, Train S01~S03 / Val S04 / Test S05, Leakage 검사 통과)

## 11. 오늘의 핵심

- 자동 QA 의 CHECK 는 탈락이 아니라 "사람이 볼 후보" — 이번 CHECK 5개는 모두 배경 물체 오검출이었다.
- 숫자(single_person_ratio)만으로 원인을 정하지 않고 원본 장면을 확인해야 한다.
- 평가용 데이터는 사람 기준으로 나눠야 Leakage 가 없다.
