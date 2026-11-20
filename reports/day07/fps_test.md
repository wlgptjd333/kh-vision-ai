# Day 07 Processing FPS Check

## Saved Video

사용한 영상: `S02_A003_T01.mp4` (walk_turn_walk, 640×480, 25 FPS, 241 Frame), `S02_A006_T01.mp4` (wave, 640×480, 94 Frame)

평균 Processing FPS: 프로그램 끝의 Overall processing FPS(처리한 Frame 수 ÷ 전체 시간) 기준 walk 41.5~43.5, wave 25~30 (RTX 3060, WSL). 43번 표의 Frame별 평균(`mean_processing_fps`)은 70~79로 더 높게 나옴. Overall은 첫 Frame의 모델 준비 시간까지 전체 시간에 들어가서 낮아지는 것으로 보임.

느리다고 느껴졌는가: 아니오. 원본 영상 25 FPS보다 빠르게 처리됨. wave의 Overall이 더 낮은 것은 영상이 짧아(94 Frame) 시작 준비 시간 비중이 커서로 보임.


## IP Webcam

Stream Resolution:

평균 Processing FPS:

영상 자체의 지연이 느껴졌는가:

Prediction 시작이 늦다고 느껴졌는가:


## 원인을 구분해 봅니다

카메라·네트워크 문제로 보이는가:

AI Processing 문제로 보이는가:

Sliding Window 때문에 기다린 것으로 보이는가:


## 한 줄 정리

-
