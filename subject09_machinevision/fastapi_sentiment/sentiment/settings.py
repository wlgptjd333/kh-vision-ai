import os
from pathlib import Path

from dotenv import load_dotenv

# Redis 주소는 코드에 직접 쓰지 않고 프로젝트 루트의 .env에서 읽습니다.
# .env는 .gitignore로 GitHub에 올리지 않고, 형식은 .env.example을 봅니다.
# Celery Worker는 db.py보다 이 파일을 먼저 읽을 수 있으므로 여기서도 .env를 불러옵니다.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f".env의 {name}을 설정해 주세요. (.env.example 참고)")
    return value


# Redis DB /1 캐시, /2 Celery Broker, /3 Celery 결과
REDIS_CACHE_URL = _require("REDIS_CACHE_URL")
CELERY_BROKER_URL = _require("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = _require("CELERY_RESULT_BACKEND")
CACHE_SECONDS = int(os.getenv("CACHE_SECONDS", "600"))
