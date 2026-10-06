import sqlite3
from swarm.workspace import Workspace


def test_workspace(tmp_path):
    w = Workspace(tmp_path / 'x.sqlite').start_run('obj')
    w.board('agent-1', 'finding', 'x'); aid = w.artifact('agent-1', 'a.txt', 'hello'); w.access('agent-2', aid)
    s = w.snapshot(); assert len(s['board']) == 1 and len(s['artifacts']) == 1 and len(s['accesses']) == 1


def test_runs_are_isolated(tmp_path):
    base = Workspace(tmp_path / 'x.sqlite')
    a, b = base.start_run('first'), base.start_run('second')
    aid = a.artifact('agent-1', 'n', 'c')
    a.board('agent-1', 'note', 'hi')
    assert b.snapshot()['artifacts'] == [] and b.recent_board() == []
    assert b.get_artifact(aid) is None and b.access('agent-2', aid) is None
    assert len(base.runs()) == 2


def test_access_validation_and_cross_agent_flag(tmp_path):
    w = Workspace(tmp_path / 'x.sqlite').start_run('o')
    aid = w.artifact('agent-1', 'n', 'content')
    assert w.access('agent-1', 999) is None           # missing id: not logged
    assert w.access('agent-1', aid)[5] == 'content'   # self read
    w.access('agent-2', aid)                          # cross-agent read
    assert [r[5] for r in w.snapshot()['accesses']] == [0, 1]


def test_non_string_content_is_serialised(tmp_path):
    w = Workspace(tmp_path / 'x.sqlite').start_run('o')
    aid = w.artifact('agent-1', {'n': 1}, ['a', 'b'])
    assert w.get_artifact(aid)[5] == '["a", "b"]'


def test_migrates_v1_schema(tmp_path):
    p = tmp_path / 'old.sqlite'
    db = sqlite3.connect(p)
    db.executescript('''CREATE TABLE board(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, agent TEXT, kind TEXT, content TEXT);
                        CREATE TABLE artifacts(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, agent TEXT, name TEXT, content TEXT);
                        INSERT INTO board(ts,agent,kind,content) VALUES(0,'a','k','old');''')
    db.commit(); db.close()
    w = Workspace(p).start_run('o')
    w.artifact('agent-1', 'n', 'c')
    assert len(w.snapshot()['artifacts']) == 1 and w.recent_board() == []   # old untagged rows don't leak in


def test_timestamps_strictly_increase_across_views_and_tables(tmp_path):
    base = Workspace(tmp_path / 'x.sqlite')
    a, b = base.start_run('a'), base.start_run('b')
    for i in range(200):
        (a if i % 2 else b).trace('agent-1', 'e', {'i': i})
        (b if i % 2 else a).board('agent-1', 'k', str(i))
    stamps = [r[1] for r in a.snapshot()['traces'] + a.snapshot()['board'] + b.snapshot()['traces'] + b.snapshot()['board']]
    assert len(stamps) == len(set(stamps)) == 400
