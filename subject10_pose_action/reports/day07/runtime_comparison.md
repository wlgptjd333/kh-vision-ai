# Day 07 Runtime Check

## 기본 Runtime 설정

Window Seconds: 5.0

Confidence Threshold: 0.55

Stabilization Buffer: 5


## 저장영상에서 확인한 내용

사용한 영상: `data/day05/videos/S02/` 6개 중 `S02_A006_T01.mp4`(wave, 3.76초)와 `S02_A003_T01.mp4`(walk_turn_walk, 9.64초). 41번 검사에서 Runtime-ready는 5 / 6이었고, 4초 Warm-up보다 짧은 wave만 0이었음.

Prediction 발생 여부:

| 실행 | 바꾼 값 | Prediction | 평균 Confidence | Raw 변화 | Stable 변화 |
|---|---|---|---|---|---|
| wave, 기본 | - | 0회 (영상이 끝날 때까지 WARMING_UP) | - | - | - |
| wave | Window 3초 | 5회 | 0.2493 | 2 | 0 |
| wave | Window 2초 | 8회 | 0.2600 | 2 | 0 |
| walk_turn_walk, 기본 | - | 21회 | 0.2305 | 5 | 0 |
| walk_turn_walk | Confidence 0.40 / 0.70 | 21회 | 0.2305 | 5 | 0 |
| walk_turn_walk | Buffer 3 / 7 | 21회 | 0.2305 | 5 | 0 |

UNKNOWN이 많았는가: 7개 실행 모두 `unknown_ratio = 1.0`. 최고 Confidence도 walk 0.263, wave 0.317이라 Threshold 0.40에도 못 미침.

관찰한 특징:

- Raw Label을 세어 보면 wave Window 2초는 8회 중 5회, 3초는 5회 중 4회가 `wave`였음. 모델이 방향은 맞히지만 확신(Confidence)이 낮아서 Threshold에서 모두 UNKNOWN이 됨.
- walk_turn_walk는 21회 중 `drink_return` 14회, `leg_raise_lower` 5회, `walk_turn_walk`는 1회뿐. 몸을 돌리는 구간에서 어깨 폭이 줄어 정규화 값이 흔들리는 문제(4일차에 본 현상)와, Train이 S01 6개 Clip뿐인 점이 함께 영향을 준 것으로 보임.
- Confidence Threshold와 Stabilization Buffer는 Random Forest 뒤에 붙는 단계라서 바꿔도 Prediction 횟수·Raw Label·Confidence는 그대로였음. 들어오는 값이 모두 UNKNOWN이라 Buffer 크기에 따른 Stable 차이도 볼 수 없었음.
- Window를 줄이면 Warm-up이 짧아져 Prediction이 빨리, 더 많이 나옴(3초 5회 → 2초 8회). 대신 Window가 짧으면 동작 전체가 아니라 일부만 보고 판단함.
- 교재 결과와 거의 같음(walk 21회·0.2305, wave 2초 8회·0.2600). wave 3초 평균 Confidence만 교재 0.2487과 0.0006 차이.


## IP Webcam에서 확인한 내용

사용한 Action:

Raw Prediction:

Confidence 경향:

Threshold 결과:

Stable 결과:


## 저장영상과 IP Webcam의 차이

-


## 한 줄 정리

-
