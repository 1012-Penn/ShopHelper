"""SQLAlchemy 声明式基类；独立放置，避免业务模型与执行层模型循环导入。"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
