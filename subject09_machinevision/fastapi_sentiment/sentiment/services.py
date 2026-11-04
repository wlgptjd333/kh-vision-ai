import hashlib
import json
from redis import Redis
from .hf_model import MODEL_ID, get_classifier, normalize_prediction
from .settings import REDIS_CACHE_URL, CACHE_SECONDS

# Redis 연결은 Python 프로세스별로 생성합니다. 서버 실행시 연결을 강제하지 않습니다.
redis_cache = Redis.from_url(REDIS_CACHE_URL, decode_responses=True, socket_connect_timeout=3)

def cache_key(text: str) -> str:
    key_input = (MODEL_ID + "\n" + text.strip()).encode("utf-8")
    return "sentiment:analysis:" + hashlib.sha256(key_input).hexdigest()

def analyze_text(text: str, use_cache: bool = False) -> dict:
    """동기 API에선 use_cache=False, Celery 워커에선 True로 실행합니다."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("빈 문장은 분석할 수 없습니다.")
    key = cache_key(cleaned)
    if use_cache:
        cached = redis_cache.get(key)
        if cached is not None:
            return json.loads(cached)
    classifier = get_classifier()
    output = classifier(cleaned, truncation=True, max_length=512)
    result = normalize_prediction(output)
    if use_cache:
        redis_cache.setex(key, CACHE_SECONDS, json.dumps(result, ensure_ascii=False))
    return result
