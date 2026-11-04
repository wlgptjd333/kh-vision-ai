# Day10 Portfolio Summary

## 프로젝트

Machine Vision Inspection Service

## 문제

기존 머신비전 검사 결과는 Python Terminal 또는 파일 중심으로 확인했다.

## 목표

기존 검사 기능을 Service Core와 FastAPI로 분리하고 Web Dashboard에서 사용할 수 있도록 연결한다.

## Reused Inspection

- Inspection Module: Day06 OCR
- Rule Version: day06-ocr-v1.0

## Service

- Service Version: day10-service-v1.0
- API: FastAPI
- Client: Web Browser (PC에서만 확인, Mobile·Tablet 접속은 시험하지 않음)

## 주요 결과

- Product Sample 선택
- OCR Text 확인
- Expected Value 비교
- Final OK / NG
- NG Reason (`NG_EXPECTED_MISMATCH`)
- Rule / Service Version
- History
- OK / NG Stats

## Architecture

`reports/day10/portfolio_architecture.md` 참고

## 현재 한계

Server에 저장된 Sample 이미지를 선택하여 검사한다. 6일차 환경 변화 테스트의 Blur·Tilt·Glare 실패는 그대로 남아 있다.

## 다음 확장

산업용 Camera, 실시간 Frame, Jetson, AI Detection, Machine Vision Rule, Database 또는 MES 연동 구조로 확장할 수 있다.
