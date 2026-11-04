# FastAPI Sentiment

Hugging Face 감정분석 모델을 FastAPI에 연결하는 교육 실습 (교재 `FastAPI_HuggingFace/LLM` 02 → 04 → 06 → 06-20).

| 태그 | 내용 |
|---|---|
| `step-02` | `/predict/`: 입력 → 모델 추론 → 화면 |
| `step-04` | `/new/`, `/items/`: SQLite 저장, 목록·상세·수정·삭제 |
| `step-06` | Redis·Celery로 `/new/` 분석을 Worker에서 처리, 결과 캐시 |
| `step-06-pg` | 저장소를 SQLite → Docker PostgreSQL(5433)로 변경, DBeaver로 확인 |

## 실행 (터미널 3개)

Docker Desktop을 먼저 켠다. `docker compose up -d`가 Redis(6380)와 PostgreSQL(5433)을 함께 띄운다.
DB·Redis 주소와 PostgreSQL 계정은 `.env`에서 읽는다 (Git에 올리지 않음). 처음이면:

```bash
cp .env.example .env   # 값 채우기
```

`.env`를 바꾸면 FastAPI와 Worker를 둘 다 다시 켠다.

```bash
cd ~/fastapi_sentiment && docker compose up -d
```

```bash
cd ~/fastapi_sentiment && source .venv/bin/activate && celery -A sentiment.celery_app:celery_app worker --loglevel=info --pool=solo
```

```bash
cd ~/fastapi_sentiment && source .venv/bin/activate && python -m uvicorn main:app --reload
```

- 동기: http://127.0.0.1:8000/predict/
- 비동기: http://127.0.0.1:8000/new/
- 목록: http://127.0.0.1:8000/items/

## 테스트

```bash
DATABASE_URL=sqlite:////tmp/test_sentiment.db python -m pytest -q
```

테스트는 DB에 글을 쓰고 지운다. 그냥 돌리면 `.env`의 PostgreSQL 실습 데이터에 섞이므로 위처럼 임시 SQLite로 돌린다.

## DB 확인

- 터미널: `docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT * FROM text_items"'`
- DBeaver: PostgreSQL, Host `127.0.0.1`, Port `5433`, DB·사용자·비밀번호는 `.env` 값
- PostgreSQL 전환 전 SQLite DB는 `backups/sentiment_before_postgresql.db`에 있다 (Git 제외)

## 다른 컴퓨터에서 이어서 작업할 때

GitHub에는 코드만 있다. 아래는 Git에 올리지 않으므로 clone만으로는 실행되지 않는다.

| 없는 것 | 해결 |
|---|---|
| `.env` (Redis·PostgreSQL 주소, 계정) | `cp .env.example .env` 후 값 채우기. 실제 값은 개인 노트(비공개)에 보관 |
| `.venv` | `uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt` |
| PostgreSQL 데이터 (Docker 볼륨) | 새 PC에서는 빈 DB로 시작. `docker compose up -d` 후 FastAPI를 켜면 `text_items` 테이블이 생긴다 |
| `sentiment.db`, `backups/` | 옮기려면 직접 복사 (선택) |
| Hugging Face 모델 | 첫 실행 때 다시 내려받는다 (약 520MB) |

`.env`가 없거나 항목이 빠지면 실행할 때 "`.env`의 ○○을 설정해 주세요" 오류가 난다. `docker compose up -d`도 `POSTGRES_PASSWORD` 등이 없으면 멈춘다.

## 측정 결과 (RTX 3060)

| 항목 | 시간 |
|---|---:|
| `POST /new/` 응답 | 0.02초 |
| 첫 분석 완료 (Worker 모델 로딩) | 약 6초 |
| 같은 문장 재분석 (캐시) | 0.02초 |
| `POST /predict/` 첫 요청 / 다음 요청 | 1.4초 / 0.005초 |
