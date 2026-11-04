"""교육용: SQLite 백업의 text_items 내용을 비어 있는 PostgreSQL로 복사합니다. (교재 06-20-10)

실행: python -m sentiment.transfer_sqlite_data
주의: 기존 id는 새 DB에서 새로 발급됩니다. 운영 DB용 범용 이전 도구가 아닙니다.
"""
import sqlite3
from datetime import datetime, timezone

from sqlalchemy import func, select

from .db import BASE_DIR, SessionLocal, engine
from .models import TextItem

source_path = BASE_DIR / "backups" / "sentiment_before_postgresql.db"
if engine.dialect.name != "postgresql":
    raise RuntimeError("대상 DB는 PostgreSQL이어야 합니다.")
if not source_path.is_file():
    raise FileNotFoundError(f"SQLite 백업 파일이 없습니다: {source_path}")

with sqlite3.connect(source_path) as source:
    source.row_factory = sqlite3.Row
    rows = source.execute("SELECT * FROM text_items ORDER BY id").fetchall()
    required = {"text", "label", "score", "created_at", "is_analyzing", "status", "revision", "error_message"}
    columns = {r[1] for r in source.execute("PRAGMA table_info(text_items)")}
    if not required.issubset(columns):
        raise RuntimeError("SQLite 백업에 06 단계 컬럼이 누락되어 있습니다.")

with SessionLocal() as target:
    count = target.scalar(select(func.count()).select_from(TextItem))
    if count != 0:
        raise RuntimeError("PostgreSQL text_items에 데이터가 있어 이전을 중지합니다.")
    for row in rows:
        created = datetime.fromisoformat(row["created_at"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        target.add(TextItem(
            text=row["text"], label=row["label"], score=row["score"],
            created_at=created, is_analyzing=bool(row["is_analyzing"]),
            status=row["status"], revision=row["revision"],
            error_message=row["error_message"],
        ))
    target.commit()
    print(f"이전 완료: SQLite {len(rows)}건 → PostgreSQL {len(rows)}건")
