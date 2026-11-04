# fastapi_resnet18 — ResNet-18 이미지 분류 + FastAPI + Redis + Celery

KH정보교육원 교재 `FastAPI_HuggingFace/Vision` 01·02 실습.

| 단계 | 내용 | 상태 |
|---|---|---|
| 01 | `microsoft/resnet-18` 동기 분류 API·화면, `training/finetune.py` | 코드 작성, 원본 모델로 동작 확인 |
| 02 | Redis 캐시, PostgreSQL Job, Celery Worker, 동기·비동기 비교 | 동작 확인, 비교 산출물 작성 |
| Fine-tuning | 콩잎 3클래스 (`angular_leaf_spot`, `bean_rust`, `healthy`) | **아직 안 함** (11/6 시험에서 진행) |

## 산출물

- [동기 vs 비동기 비교](reports/sync_async_comparison.md)
- 측정 스크립트: `tools/bench_sync_async.py` (교재 외 추가)

## 실행

```bash
cd ~/fastapi_resnet18 && docker compose up -d
```

```bash
cd ~/fastapi_resnet18 && source .venv/bin/activate && celery -A ai.celery_app:celery_app worker --loglevel=info --pool=solo --concurrency=1
```

```bash
cd ~/fastapi_resnet18 && source .venv/bin/activate && uvicorn main:app --host 127.0.0.1 --port 8000
```

- 동기 화면: http://127.0.0.1:8000/vision/sync/
- 비동기 화면: http://127.0.0.1:8000/vision/async/

측정 (세 개가 모두 켜진 상태에서):

```bash
cd ~/fastapi_resnet18 && source .venv/bin/activate && python -m tools.bench_sync_async --images dataset/test --count 10
```

`dataset/`은 Git에 올리지 않는다. 콩잎 데이터는 Hugging Face `AI-Lab-Makerere/beans`.

## 환경

WSL2 Ubuntu, Python 3.11 (uv), torch 2.15.0+cu132, transformers 4.57.6. 교재대로 추론은 CPU(`device=-1`).
