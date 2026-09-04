from sqlalchemy import text

from app.config import Settings
from app.db import make_engine, make_session_factory


def test_make_engine_and_session_roundtrip():
    settings = Settings(openai_api_key="test", database_url="sqlite:///:memory:")
    engine = make_engine(settings)
    factory = make_session_factory(engine)
    with factory() as session:
        assert session.execute(text("SELECT 1")).scalar() == 1
