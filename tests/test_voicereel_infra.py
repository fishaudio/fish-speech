import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import importlib

from voicereel.flask_app import create_app
from voicereel.task_queue import TaskQueue
from voicereel.db import init_db


def test_create_app():
    app = create_app()
    assert hasattr(app, 'route')
    assert hasattr(app, 'routes')
    assert isinstance(app.routes, dict)

def test_task_queue_basic():
    q = TaskQueue()
    called = []
    def dummy(x):
        called.append(x)
    q.enqueue(dummy, 1)
    q.process_next()
    assert called == [1]

def test_init_db_tables():
    conn = init_db()
    cur = conn.cursor()
    for tbl in ['speakers', 'jobs', 'usage']:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,))
        assert cur.fetchone() is not None

def test_init_db_postgres(monkeypatch):
    calls = {}

    class DummyCursor:
        def execute(self, sql):
            calls.setdefault('exec', []).append(sql)

    class DummyConn:
        def cursor(self):
            return DummyCursor()

        def commit(self):
            calls['commit'] = True

    def fake_connect(dsn):
        calls['dsn'] = dsn
        return DummyConn()

    monkeypatch.setitem(importlib.import_module('sys').modules, 'psycopg2', type('PG', (), {'connect': staticmethod(fake_connect)}))
    conn = init_db('postgresql://example/db')
    assert isinstance(conn, DummyConn)
    assert calls.get('dsn') == 'postgresql://example/db'
    assert 'commit' in calls

