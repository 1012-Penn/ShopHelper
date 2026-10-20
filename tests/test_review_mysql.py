"""可选真实 MySQL 回归。只允许显式指定 shophelper_test_* 临时数据库。"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from pymysql.constants import CLIENT
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.flywheel import FlywheelService
from app.models import KnowledgeChunk

ROOT = Path(__file__).resolve().parents[1]


def apply_sql(engine, relative):
    connection = engine.raw_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute((ROOT / relative).read_text())
            while cursor.nextset():
                pass
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def mysql_factory():
    url = os.environ.get("SHOPHELPER_TEST_MYSQL_URL")
    if not url:
        pytest.skip("set SHOPHELPER_TEST_MYSQL_URL to an isolated MySQL test database")
    if not (make_url(url).database or "").startswith("shophelper_test_"):
        pytest.fail("MySQL regression tests require a shophelper_test_* database")
    engine = create_engine(url, connect_args={"client_flag": CLIENT.MULTI_STATEMENTS})
    # 只清理显式 opt-in 的临时测试库;生产数据库名会被上方检查拒绝。
    with engine.begin() as connection:
        connection.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for name in inspect(connection).get_table_names():
            connection.execute(text(f"DROP TABLE `{name}`"))
        connection.execute(text("SET FOREIGN_KEY_CHECKS=1"))
    apply_sql(engine, "db/init.sql")
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory, engine
    finally:
        engine.dispose()


def test_mysql_migration_backfills_smallest_id_and_enforces_unique_review(mysql_factory):
    factory, engine = mysql_factory
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO review_queue(id,normalized_question) VALUES(1,'运费'),(2,'物流')"))
        connection.execute(text("""INSERT INTO knowledge_chunks(category,questions,answer,section_path)
            VALUES('飞轮','运费','最早答案','review_queue:1'),
                  ('飞轮','运费','后来答案','review_queue:1'),
                  ('手册','物流','不关联','manual.md > 发货')"""))
    apply_sql(engine, "db/reliability-review-index.sql")
    with factory() as session:
        chunks = session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)).all()
        assert len(chunks) == 3
        assert [row.review_id for row in chunks] == [1, None, None]
        assert [row.answer for row in chunks] == ['最早答案', '后来答案', '不关联']
        session.add(KnowledgeChunk(review_id=1, category='重复', questions='q', answer='a'))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        session.add(KnowledgeChunk(review_id=999, category='孤儿', questions='q', answer='a'))
        with pytest.raises(IntegrityError):
            session.commit()


def test_mysql_concurrent_approval_uses_one_chunk(mysql_factory):
    factory, engine = mysql_factory
    apply_sql(engine, "db/reliability-review-index.sql")
    rid = FlywheelService(factory).capture('self_check', None, '运费怎么收', '', None)
    barrier = Barrier(4)

    def approve(_):
        barrier.wait(timeout=10)
        return FlywheelService(factory).approve(rid, '按目的地计算')

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(approve, range(4)))
    assert len(set(ids)) == 1
    with factory() as session:
        rows = session.scalars(select(KnowledgeChunk)).all()
        assert len(rows) == 1 and rows[0].review_id == rid
