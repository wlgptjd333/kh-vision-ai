from .celery_app import celery_app
from .db import SessionLocal
from .models import TextItem
from .services import analyze_text

@celery_app.task(bind=True, name="sentiment.tasks.analyze_textitem_task")
def analyze_textitem_task(self, item_id: int, revision: int):
    """ID로 DB 조회 → Redis 캐시/모델 → DB 갱신. 오래된 수정 작업은 무시합니다."""
    with SessionLocal() as db:
        item = db.get(TextItem, item_id)
        if item is None or item.revision != revision:
            return {"status": "ignored", "reason": "deleted_or_old_revision"}
        text = item.text
    try:
        result = analyze_text(text, use_cache=True)
    except Exception as exc:
        with SessionLocal() as db:
            current = db.get(TextItem, item_id)
            if current is not None and current.revision == revision:
                current.is_analyzing = False
                current.status = "FAILED"
                current.error_message = type(exc).__name__  # 내부 오류 세부정보는 화면에 노출하지 않습니다.
                db.commit()
        raise
    with SessionLocal() as db:
        current = db.get(TextItem, item_id)
        if current is None or current.revision != revision:
            return {"status": "ignored", "reason": "deleted_or_old_revision"}
        current.label = result["label"]
        current.score = result["score"]
        current.is_analyzing = False
        current.status = "DONE"
        current.error_message = ""
        db.commit()
    return {"status": "DONE", **result}
