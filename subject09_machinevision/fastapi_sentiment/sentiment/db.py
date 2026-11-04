# ============================================================
# ① 라이브러리 가져오기
# ============================================================
# os, Path       : .env 위치를 찾고 환경변수를 읽기 위해 사용
# load_dotenv    : .env 파일의 값을 환경변수로 불러옴
# create_engine   : 데이터베이스 연결을 관리하는 엔진 생성
# DeclarativeBase : ORM 모델의 부모 클래스
# sessionmaker    : DB 작업용 세션을 생성하는 도구

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


# ============================================================
# ② .env에서 PostgreSQL 주소 읽기 (06-20에서 SQLite → PostgreSQL로 변경)
# ============================================================
# 프로젝트 루트(fastapi_sentiment/.env)의 DATABASE_URL을 읽습니다.
#
# 형식: postgresql+psycopg://<사용자>:<비밀번호>@127.0.0.1:5433/<DB이름>  (.env.example 참고)
#
# FastAPI와 Celery Worker가 같은 .env를 읽어야
# 같은 DB를 보고 비동기 작업이 DONE까지 갑니다.
#
# 04·06에서 쓰던 SQLite 주소: sqlite:///./sentiment.db

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(".env의 DATABASE_URL을 설정해 주세요.")


# ============================================================
# ③ 데이터베이스 엔진 생성
# ============================================================
# pool_pre_ping=True
# : 연결을 꺼내 쓰기 전에 살아 있는지 확인
#   (DB 컨테이너를 재시작해도 끊긴 연결로 오류가 나지 않게)
#
# SQLite 전용 옵션 check_same_thread=False는 제거했습니다.

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# ============================================================
# ④ DB 세션을 생성하는 SessionLocal 설정
# DB 세션(Session)은 FastAPI가 데이터베이스에서 조회·저장·수정·삭제 작업을 수행할 때 
# 사용하는 작업 공간입니다.
# 쉽게 말하면 데이터베이스에 작업을 요청하고, 그 작업을 관리하는 담당자라고 생각하면 됩니다.
# 이 코드는 DB 세션 자체가 아니라, DB 세션을 생성하는 도구를 만드는 코드입니다.
# ============================================================
# sessionmaker() : 새로운 DB 세션을 만드는 팩토리 생성
#
# bind=engine
# : ③에서 생성한 DB 엔진에 연결
#
# autoflush=False
# : 쿼리 전에 변경 내용을 자동 반영하지 않도록 설정
#
# expire_on_commit=False
# : commit 이후에도 이미 읽어온 객체 속성값 유지
#
# 주의:
# 여기서는 세션 자체를 생성하는 것이 아니라
# 세션을 만드는 SessionLocal을 준비합니다.

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False
)


# ============================================================
# ⑤ ORM 모델의 공통 부모 클래스 생성
# ============================================================
# Base를 상속하여 데이터베이스 테이블 모델을 만듭니다.
#
# 예:
# class TextItem(Base):
#     ...
#
# 실제 TextItem 테이블 구조는 models.py에서 정의합니다.

class Base(DeclarativeBase):

    # 부모 클래스에 추가할 내용이 없으므로 pass 사용
    pass


# ============================================================
# ⑥ FastAPI에서 사용할 DB 세션 제공 함수
# ============================================================
# get_db() : API 요청 처리에 사용할 세션 제공
#
# router.py에서 다음과 같이 사용합니다.
#
# db: Session = Depends(get_db)
#
# FastAPI가 이 함수를 호출하여 세션을 전달합니다.

def get_db():

    # --------------------------------------------------------
    # ⑦ 새로운 데이터베이스 세션 생성
    # --------------------------------------------------------
    # SessionLocal()을 호출하여 DB 작업용 세션 생성
    #
    # with 문은 작업이 끝나면 세션을 닫도록 관리합니다.

    with SessionLocal() as db:

        # ----------------------------------------------------
        # ⑧ FastAPI 라우터에 DB 세션 전달
        # ----------------------------------------------------
        # yield : 생성한 DB 세션을 라우터에 제공
        #
        # 라우터는 이 세션을 사용하여
        # 조회, 추가, 수정, 삭제 작업을 수행합니다.

        yield db

    # --------------------------------------------------------
    # ⑨ DB 세션 종료
    # --------------------------------------------------------
    # 라우터에서 세션 사용이 끝나면
    # with 블록이 종료되면서 세션을 닫습니다.
    #
    # 단, 데이터 저장(commit)은 자동으로 하지 않습니다.
    # CRUD 코드에서 필요한 경우 db.commit()을 호출합니다.
