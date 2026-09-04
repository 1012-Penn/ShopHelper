"""数据库引擎与会话工厂——同步 SQLAlchemy,演示场景直接在 async handler 里调用。"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings


def make_engine(settings: Settings):
    return create_engine(settings.database_url, pool_pre_ping=True, pool_recycle=3600)


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
