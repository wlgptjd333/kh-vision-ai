from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sentiment.router import router as sentiment_router
from sentiment.db import Base, engine
from sentiment import models  # Base에 TextItem 테이블 등록

app = FastAPI(title="FastAPI Sentiment CRUD · PostgreSQL")

# PostgreSQL에 아직 없는 테이블을 생성합니다.
# create_all()은 없는 테이블만 만들고, 기존 테이블 구조를 바꾸거나 SQLite 데이터를 복사하지 않습니다.
# 06의 upgrade_schema()(SQLite 04→06 컬럼 추가)는 PostgreSQL에서는 실행하지 않습니다.
Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(sentiment_router)

@app.get("/health", name="health")
def health():
    return {"status": "ok"}
