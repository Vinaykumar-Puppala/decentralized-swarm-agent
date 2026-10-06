import json, sqlite3, threading, time, uuid

TABLES = {
    'runs': 'run_id TEXT PRIMARY KEY, ts REAL, objective TEXT, config TEXT, status TEXT, final_report TEXT',
    'board': 'id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, agent TEXT, kind TEXT, content TEXT',
    'artifacts': 'id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, agent TEXT, name TEXT, kind TEXT, content TEXT',
    'accesses': 'id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, reader TEXT, artifact_id INTEGER, author TEXT, cross_agent INTEGER, action TEXT',
    'traces': 'id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, agent TEXT, event TEXT, payload TEXT',
}


def _text(v):
    """SQLite only binds scalars; LLMs sometimes return dicts/lists for text fields."""
    if v is None or isinstance(v, str):
        return v
    return json.dumps(v, default=str)


class Workspace:
    """SQLite-backed shared environment. One connection guarded by a lock so agent threads can share it safely.
    Every row is tagged with a run_id so experiments never see each other's data."""

    def __init__(self, path="workspace.sqlite", run_id=None):
        self.path = str(path)
        self.run_id = run_id
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        with self._lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            for name, cols in TABLES.items():
                self.db.execute(f"CREATE TABLE IF NOT EXISTS {name}({cols})")
            self._migrate()
            self.db.commit()

    def _migrate(self):
        # Databases created by the original V1 schema lack the newer columns.
        for name, cols in TABLES.items():
            have = {r[1] for r in self.db.execute(f"PRAGMA table_info({name})")}
            for col in cols.split(','):
                col_name, col_type = col.split()[:2]
                if col_name not in have:
                    self.db.execute(f"ALTER TABLE {name} ADD COLUMN {col_name} {col_type}")

    def close(self):
        with self._lock:
            self.db.close()

    def _exec(self, sql, params=()):
        with self._lock:
            cur = self.db.execute(sql, params)
            self.db.commit()
            return cur.lastrowid

    def _q(self, sql, params=()):
        with self._lock:
            return self.db.execute(sql, params).fetchall()

    def for_run(self, run_id):
        """A view of the same database bound to a different run."""
        w = Workspace.__new__(Workspace)
        w.__dict__.update(self.__dict__)
        w.run_id = run_id
        return w

    # ---- runs
    def start_run(self, objective, config=None):
        run_id = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
        self._exec("INSERT INTO runs(run_id,ts,objective,config,status) VALUES(?,?,?,?,?)",
                   (run_id, time.time(), objective, json.dumps(config or {}, default=str), 'running'))
        return self.for_run(run_id)

    def finish_run(self, status='finished', final_report=None):
        self._exec("UPDATE runs SET status=?, final_report=COALESCE(?, final_report) WHERE run_id=?",
                   (status, _text(final_report), self.run_id))

    def runs(self):
        return self._q("SELECT run_id,ts,objective,status FROM runs ORDER BY ts DESC")

    def run_info(self):
        r = self._q("SELECT run_id,ts,objective,config,status,final_report FROM runs WHERE run_id=?", (self.run_id,))
        return r[0] if r else None

    # ---- writes
    def board(self, agent, kind, content):
        return self._exec("INSERT INTO board(run_id,ts,agent,kind,content) VALUES(?,?,?,?,?)",
                          (self.run_id, time.time(), agent, kind, _text(content)))

    def artifact(self, agent, name, content, kind='finding'):
        return self._exec("INSERT INTO artifacts(run_id,ts,agent,name,kind,content) VALUES(?,?,?,?,?,?)",
                          (self.run_id, time.time(), agent, _text(name), kind, _text(content)))

    def access(self, reader, artifact_id, action="read"):
        """Log a read and return the artifact row, or None if it doesn't exist in this run."""
        art = self.get_artifact(artifact_id)
        if art is None:
            return None
        author = art[2]
        self._exec("INSERT INTO accesses(run_id,ts,reader,artifact_id,author,cross_agent,action) VALUES(?,?,?,?,?,?,?)",
                   (self.run_id, time.time(), reader, art[0], author, int(author != reader), action))
        return art

    def trace(self, agent, event, payload):
        return self._exec("INSERT INTO traces(run_id,ts,agent,event,payload) VALUES(?,?,?,?,?)",
                          (self.run_id, time.time(), agent, event, json.dumps(payload, default=str)))

    # ---- reads
    def get_artifact(self, artifact_id):
        try:
            artifact_id = int(artifact_id)
        except (TypeError, ValueError):
            return None
        r = self._q("SELECT id,ts,agent,name,kind,content FROM artifacts WHERE id=? AND run_id IS ?", (artifact_id, self.run_id))
        return r[0] if r else None

    def recent_board(self, limit=30):
        rows = self._q("SELECT id,agent,kind,content FROM board WHERE run_id IS ? ORDER BY id DESC LIMIT ?", (self.run_id, limit))
        return rows[::-1]

    def artifact_index(self, limit=40, preview=160):
        """Public listing: titles and short previews only. Full content requires an explicit read."""
        rows = self._q("SELECT id,agent,name,kind,substr(content,1,?) FROM artifacts WHERE run_id IS ? ORDER BY id DESC LIMIT ?",
                       (preview, self.run_id, limit))
        return rows[::-1]

    def snapshot(self):
        q = lambda t, cols: self._q(f"SELECT {cols} FROM {t} WHERE run_id IS ? ORDER BY id", (self.run_id,))
        return {"board": q('board', 'id,ts,agent,kind,content'),
                "artifacts": q('artifacts', 'id,ts,agent,name,kind,content'),
                "accesses": q('accesses', 'id,ts,reader,artifact_id,author,cross_agent,action'),
                "traces": q('traces', 'id,ts,agent,event,payload')}
