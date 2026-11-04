"""04단계 SQLite 테이블을 06단계로 확장합니다 (교육용, 반복 실행 가능)."""
from sqlalchemy import inspect, text
from .db import Base, engine
from . import models

NEW_COLUMNS = {
    'is_analyzing': 'BOOLEAN NOT NULL DEFAULT 0',
    'status': "VARCHAR(20) NOT NULL DEFAULT 'DONE'",
    'revision': 'INTEGER NOT NULL DEFAULT 1',
    'error_message': "VARCHAR(255) NOT NULL DEFAULT ''",
}

def upgrade_schema():
    Base.metadata.create_all(bind=engine)
    existing = {col['name'] for col in inspect(engine).get_columns('text_items')}
    with engine.begin() as connection:
        for column, definition in NEW_COLUMNS.items():
            if column not in existing:
                connection.execute(text(f'ALTER TABLE text_items ADD COLUMN {column} {definition}'))
                print(f'added: {column}')
    print('06 DB schema ready')

if __name__ == '__main__':
    upgrade_schema()
