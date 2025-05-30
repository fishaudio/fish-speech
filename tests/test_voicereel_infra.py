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
