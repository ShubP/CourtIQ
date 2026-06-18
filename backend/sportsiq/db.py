"""Database engine/session setup. Driven entirely by DATABASE_URL."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import DATABASE_URL
from .models import Base

# SQLite needs check_same_thread=False across FastAPI threads, and a busy
# timeout so the API (reader) and pipeline (writer) don't trip over locks.
connect_args = (
    {"check_same_thread": False, "timeout": 30}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Session:
    """Transactional session context manager."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
