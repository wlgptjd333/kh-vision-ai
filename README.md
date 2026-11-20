# product-review-service

상품 리뷰 서비스 실습 — **Django + DRF 서비스 서버(backend)** 와 **FastAPI AI 서버(ai-server)** 를 나눠서 만든다.

```
Users → Clients → Backend Server (Django·DRF) ─┬→ Static File (static/·media/)
Admin → Clients → Admin Server (Django Admin)   ├→ Database (SQLite → PostgreSQL)
ML    → Clients → ML Server (FastAPI)  ←── ai_gateway가 HTTP로 호출
```

**왜 나누나:** AI 모델은 무겁고 웹 서버와 바뀌는 주기가 다르다. 한 프로세스에 넣으면 모델 로딩이 웹 응답을 막고 패키지 버전도 꼬인다. 그래서 서버마다 폴더와 `.venv`를 따로 둔다.

## 폴더

| 폴더 | 내용 |
|---|---|
| `backend/` | Django 프로젝트 `mysite` + 앱 5개 (`apps/accounts`·`products`·`reviews`·`interactions`·`ai_gateway`) |
| `ai-server/` | FastAPI AI 서버 (13단계부터) |

## 진행

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 프로젝트 기본 구축: 폴더·`.venv`·패키지, startproject, 앱 5개, Custom User, pre-commit | ✅ |
| 2~ | CRUD · 모델 · DRF · JWT · PostgreSQL · Celery·Redis · Docker · WebSocket | 수업 진행에 맞춰 |

## 실행 (WSL Ubuntu)

```bash
cd backend
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

- Python 3.12, Django 6.1, DRF 3.18
- 앱을 `apps/` 아래 두었기 때문에 각 `apps.py`의 `name`은 `"apps.<앱이름>"`이고 `INSTALLED_APPS`도 같은 경로로 등록한다.
- `AUTH_USER_MODEL = "accounts.User"` 는 첫 migrate 전에 정해야 한다 (나중에 바꾸면 DB를 다시 만들어야 함).
- `db.sqlite3`, `media/`, `.env`, `.venv` 는 올리지 않는다.
