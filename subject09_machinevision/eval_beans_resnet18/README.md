# 교과 9 평가과제 — 콩잎 질병 분류 ResNet-18 동기·비동기 서빙

> 진행 중. 제출 전까지 측정 결과와 증빙은 올리지 않습니다.

## 과제 요약

- 데이터: 제공 콩잎 이미지 `angular_leaf_spot` / `bean_rust` / `healthy` (원본 `train`·`validation`·`test`)
- 모델: `microsoft/resnet-18` → 3클래스 Head-only Fine-tuning → `artifacts/beans_resnet18/`
- 서빙: FastAPI 동기 API + Celery 비동기 작업, Redis(캐시·브로커) + PostgreSQL(`image_jobs`)
- 제출: `reports/` 산출물 1~8 + 이 README

## 진행 순서

- [ ] STEP 1 데이터 확인·배치 (`validation` → `dataset/val`), Head-only 학습
- [ ] STEP 2 동기 API 검증 (`/vision/sync/`)
- [ ] STEP 3 비동기 검증 (Docker Redis·PostgreSQL, Celery Worker, `image_jobs` 조회)
- [ ] STEP 4 성능 측정(`reports/data/timings.csv`), 기능·오류 테스트, 산출물 작성

## 산출물

| 번호 | 파일 | 상태 |
|---|---|---|
| 1 | `reports/01_FineTuning_결과보고서.md` | 미작성 |
| 2 | `reports/02_시스템_아키텍처.md` | 미작성 |
| 3 | `reports/03_API_명세서.md` | 미작성 |
| 4 | `reports/04_DB_구조_작업이력.md` | 미작성 |
| 5 | `reports/05_AI버튼_처리흐름.md` | 미작성 |
| 6 | `reports/06_성능비교_보고서.md` | 미작성 |
| 7 | `reports/07_기능테스트_오류분석.md` | 미작성 |
| 8 | `reports/08_최종기술보고서.md` | 미작성 |

증빙 캡처는 `reports/evidence/`, 도표는 `reports/figures/`, 측정 원본은 `reports/data/`에 둡니다.

## Git에 올리지 않는 것

`dataset/`(제공 데이터), `artifacts/`(학습 모델), `media/`(업로드 이미지), `.env`
