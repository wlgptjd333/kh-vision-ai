# ============================================================
# ① Celery 라이브러리 가져오기
# ============================================================
# Celery : 시간이 걸리는 작업을 별도 Worker에서
#          비동기적으로 처리할 수 있도록 도와주는 도구

from celery import Celery


# ============================================================
# ② Redis 연결 주소 가져오기
# ============================================================
# settings.py에서 미리 설정한 Redis 주소를 가져옵니다.
#
# CELERY_BROKER_URL
# : 처리할 작업을 전달하는 Redis 주소 (DB 2)
#
# CELERY_RESULT_BACKEND
# : 작업 상태와 결과를 저장하는 Redis 주소 (DB 3)

from .settings import CELERY_BROKER_URL, CELERY_RESULT_BACKEND


# ============================================================
# ③ Celery 애플리케이션 생성
# ============================================================
# Celery("sentiment") : Celery 애플리케이션 이름 지정
#
# broker  : 비동기 작업을 전달할 Redis 연결 주소
# backend : 작업 상태와 결과를 저장할 Redis 연결 주소
# include : Worker가 불러올 작업 함수 파일
#
# sentiment.tasks는 이후 작성할 tasks.py를 의미합니다.
#
# 주의:
# 여기서는 Celery 실행 환경만 설정합니다.
# 실제 Worker는 별도 명령어로 실행해야 합니다.

celery_app = Celery(
    "sentiment",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["sentiment.tasks"],
)


# ============================================================
# ④ 작업 데이터와 결과의 형식 설정
# ============================================================
# conf.update() : Celery의 세부 설정을 적용하는 메서드
#
# task_serializer="json"
# : 작업을 전달할 때 데이터를 JSON으로 변환
#
# accept_content=["json"]
# : JSON 형식의 작업 메시지만 허용
#
# result_serializer="json"
# : 작업 결과도 JSON으로 변환하여 저장

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # --------------------------------------------------------
    # ⑤ Celery 시간대 설정
    # --------------------------------------------------------
    # Asia/Seoul : 한국 시간대 설정
    #
    # 주로 작업 스케줄 등의 시간 처리에 사용합니다.

    timezone="Asia/Seoul",

    # --------------------------------------------------------
    # ⑥ 작업 시작 상태 추적
    # --------------------------------------------------------
    # task_track_started=True
    #
    # Worker가 작업을 시작하면
    # STARTED 상태를 기록할 수 있도록 설정합니다.
    #
    # 예:
    # PENDING → STARTED → SUCCESS

    task_track_started=True,

    # --------------------------------------------------------
    # ⑦ Redis 연결 실패 시 재시도 설정
    # --------------------------------------------------------
    # broker_connection_retry_on_startup=True
    #
    # Celery Worker 시작 시 Redis에 연결하지 못하면
    # 연결을 다시 시도하도록 설정합니다.
    #
    # 재시도 횟수 등은 Celery의 다른 설정에도 영향을 받습니다.

    broker_connection_retry_on_startup=True,
)
