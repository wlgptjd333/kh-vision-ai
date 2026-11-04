# Day07 Failure Analysis

## Test FP

File: `samples/day07/test/normal/015.png`

Score: 0.070634 (Threshold 0.067458)

원인 추정: 파손·오염이 없는 정상 병인데 Threshold를 조금 넘었다. Validation 정상 42장의 최댓값도 0.0708이어서 정상 Score 범위의 윗부분과 겹친다. 95 Percentile 기준은 정상의 약 5%가 넘도록 정한 값이므로 이런 FP가 일부 생길 수 있다.

## Test FN

File: `samples/day07/test/anomaly/broken_small/000.png`

Score: 0.053315

원인 추정: Heatmap의 가장 뜨거운 곳은 Ground Truth의 파손 위치와 맞는다. 그런데 Score는 이미지 전체 Feature를 평균한 벡터로 계산해서 작은 파손의 차이가 묽어진 것으로 보인다. 위치는 찾았지만 판정 Score가 낮았다.

## Challenge FP

변형 조건: Dark

Score: 0.123438

원인 추정: 제품은 그대로인데 밝기만 달라졌다. 이 Baseline은 "정상 Train과 얼마나 다르게 보이는가"만 재므로 조명 변화도 이상과 구분하지 못한다. 같은 Score로 비교하면 broken_large 평균(0.1690)보다는 낮지만 contamination 평균(0.1161)보다 높다.

## 한 줄 정리

현재 Baseline이 잘 동작한 경우: broken_large, contamination (Test 정확도 1.0)

현재 Baseline이 약했던 경우: 작은 파손(broken_small 22장 중 4장 FN), 촬영 조건 변화(Dark, Bright, Shift FP)

다음에 개선하고 싶은 점: 이미지 전체 평균 대신 위치별 Score의 최댓값으로 판정하는 방법을 Validation에서 비교한다. 조명·위치가 다른 정상 이미지를 Normal Train에 추가하고, 새 버전으로 Freeze한 뒤 새 Test로 확인한다.
