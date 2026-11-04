# ============================================================
# ① [04 유지] 날짜와 시간 처리 라이브러리 가져오기
# ============================================================
# datetime : 날짜와 시간을 표현하는 클래스
# timezone : 시간대 정보 설정
#
# timezone.utc : UTC(세계 표준시) 사용

from datetime import datetime, timezone


# ============================================================
# ② [06 수정] SQLAlchemy 컬럼 타입 가져오기
# ============================================================
# [04 기존]
# DateTime : 날짜 및 시간
# Float    : 실수
# Integer  : 정수
# String   : 문자열
# Text     : 긴 문자열
#
# [06 신규]
# Boolean  : 참(True) / 거짓(False) 저장
#
# 비동기 AI 분석이 진행 중인지 기록하기 위해
# Boolean 타입을 추가했습니다.

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    Boolean
)


# ============================================================
# ③ [04 유지] ORM 모델 관련 기능 가져오기
# ============================================================
# Mapped        : Python 속성과 DB 컬럼의 타입 연결
# mapped_column : 실제 DB 컬럼의 상세 설정

from sqlalchemy.orm import Mapped, mapped_column


# ============================================================
# ④ [04 유지] 공통 ORM 부모 클래스 가져오기
# ============================================================
# db.py에서 정의한 Base를 가져옵니다.
#
# Base를 상속하면 SQLAlchemy ORM 모델을 만들 수 있습니다.

from .db import Base


# ============================================================
# ⑤ [04 유지] TextItem 모델과 테이블 이름 정의
# ============================================================
# TextItem : 감정분석 데이터를 관리하는 ORM 모델
#
# __tablename__ = "text_items"
# : 실제 SQLite에 생성되는 테이블 이름
#
# 06에서도 기존 text_items 테이블을 그대로 사용합니다.
# 새 테이블을 만드는 것이 아니라 컬럼을 추가합니다.

class TextItem(Base):

    __tablename__ = "text_items"


    # --------------------------------------------------------
    # ⑥ [04 유지] id : 데이터 고유번호
    # --------------------------------------------------------
    # Integer          : 정수형
    # primary_key=True : 기본키(PK)
    #
    # 각 감정분석 데이터를 구별하는 고유번호입니다.

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )


    # --------------------------------------------------------
    # ⑦ [04 유지] text : 사용자가 입력한 문장
    # --------------------------------------------------------
    # Text           : 긴 문자열 저장
    # nullable=False : NULL 저장 불가
    #
    # 예: "오늘 정말 기분이 좋아요."

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )


    # --------------------------------------------------------
    # ⑧ [04 유지] label : AI 감정분석 결과
    # --------------------------------------------------------
    # String(50)     : 문자열 컬럼
    # default=""     : 기본값은 빈 문자열
    # nullable=False : NULL 저장 불가
    #
    # 예: Positive, Negative, Neutral
    #
    # 06에서는 AI 분석이 완료되기 전까지
    # label이 빈 문자열일 수 있습니다.

    label: Mapped[str] = mapped_column(
        String(50),
        default="",
        nullable=False
    )


    # --------------------------------------------------------
    # ⑨ [04 유지] score : AI 감정분석 점수
    # --------------------------------------------------------
    # Float         : 실수형
    # float | None  : 실수 또는 None
    # nullable=True : NULL 허용
    #
    # 예: 0.95
    #
    # 06에서는 AI 분석이 완료되기 전까지
    # score가 None일 수 있습니다.

    score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )


    # --------------------------------------------------------
    # ⑩ [04 유지] created_at : 데이터 생성 시각
    # --------------------------------------------------------
    # DateTime(timezone=True) : 날짜와 시간 타입
    # datetime.now(timezone.utc) : 현재 UTC 시각
    #
    # 데이터가 처음 저장되는 시각을 기록합니다.

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


    # --------------------------------------------------------
    # ⑪ [06 신규] is_analyzing : AI 분석 진행 여부
    # --------------------------------------------------------
    # Boolean       : True 또는 False
    # default=False : 기본값은 False
    # nullable=False: NULL 저장 불가
    #
    # True  : AI 분석 작업이 진행 중이거나 대기 중
    # False : 현재 분석 중인 작업이 없음
    #
    # 06의 router.py에서는 비동기 작업 등록 시
    # True로 설정하고, 작업 완료 시 False로 변경합니다.

    is_analyzing: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )


    # --------------------------------------------------------
    # ⑫ [06 신규] status : AI 작업의 처리 상태
    # --------------------------------------------------------
    # String(20)    : 상태 문자열 저장
    # default="DONE": 기본값은 DONE
    #
    # 예:
    # PENDING : 작업 등록 후 처리 대기
    # DONE    : AI 분석 완료
    # FAILED  : AI 분석 실패
    #
    # 초기값 DONE은 기존 동기식 완료 데이터와
    # 호환하기 위한 기본 설정입니다.
    #
    # 새 비동기 작업에서는 router.py에서
    # PENDING 상태를 명시적으로 설정합니다.

    status: Mapped[str] = mapped_column(
        String(20),
        default="DONE",
        nullable=False
    )


    # --------------------------------------------------------
    # ⑬ [06 신규] revision : 데이터 수정 버전
    # --------------------------------------------------------
    # Integer       : 정수형
    # default=1     : 최초 버전은 1
    # nullable=False : NULL 저장 불가
    #
    # 예:
    # 최초 작성 → revision = 1
    # 첫 수정   → revision = 2
    # 두 번째 수정 → revision = 3
    #
    # 비동기 AI 작업이 여러 개 발생할 때
    # 이전 작업의 결과가 최신 데이터를 덮어쓰지 않도록
    # 버전을 비교하는 데 사용할 수 있습니다.
    #
    # 실제 버전 증가와 비교는 router.py와
    # tasks.py의 코드에서 처리해야 합니다.

    revision: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False
    )


    # --------------------------------------------------------
    # ⑭ [06 신규] error_message : AI 분석 오류 메시지
    # --------------------------------------------------------
    # String(255)   : 오류 내용을 저장할 문자열 컬럼
    # default=""    : 기본값은 빈 문자열
    # nullable=False : NULL 저장 불가
    #
    # AI 분석이 실패했을 때 오류 내용을 저장합니다.
    #
    # 예: "모델 추론 중 오류가 발생했습니다."
    #
    # 정상적으로 처리된 경우에는 빈 문자열을 사용합니다.

    error_message: Mapped[str] = mapped_column(
        String(255),
        default="",
        nullable=False
    )
