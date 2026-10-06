import json, sqlite3, threading, time, uuid

TABLES = {
    'runs': 'run_id TEXT PRIMARY KEY, ts REAL, objective TEXT, config TEXT, status TEXT, final_report TEXT',
    'board': 'id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, agent TEXT, kind TEXT, content TEXT',
    'artifacts': 'id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, agent TEXT, name TEXT, kind TEXT, content TEXT',
    'accesses': 'id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, reader TEXT, artifact_id INTEGER, author TEXT, cross_agent INTEGER, action TEXT',
    'traces': 'id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, agent TEXT, event TEXT, payload TEXT',
    # which data rows each agent has looked at, stored as inclusive id ranges: [[start, end], ...]
    'row_views': 'id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, agent TEXT, tbl TEXT, ranges TEXT',
}


def to_ranges(ids):
    """[3,4,5,9] -> [[3,5],[9,9]]"""
    out = []
    for i in sorted(set(int(x) for x in ids)):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def merge_ranges(ranges):
    return to_ranges(i for a, b in ranges for i in range(a, b + 1))


def missing_ranges(covered, total):
    """Ranges of row ids in [0, total) not in `covered` (a list of inclusive ranges)."""
    gaps, nxt = [], 0
    for a, b in merge_ranges(covered):
        if a > nxt:
            gaps.append([nxt, min(a - 1, total - 1)])
        nxt = max(nxt, b + 1)
    if nxt < total:
        gaps.append([nxt, total - 1])
    return [g for g in gaps if g[0] < total]


READ_COLS = {   # traces first: a trace can reference rows (artifacts) written just before it, which are then read after it
    'traces': 'id,ts,agent,event,payload',
    'board': 'id,ts,agent,kind,content',
    'artifacts': 'id,ts,agent,name,kind,content',
    'accesses': 'id,ts,reader,artifact_id,author,cross_agent,action',
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
        self._clock = [0.0]          # shared by reference with for_run() views: last timestamp handed out
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

    def _ts(self):
        """Strictly increasing wall-clock timestamp. Windows' clock ticks only every ~15 ms, so plain time.time() gives
        many rows the same value and the causal order of events (call before result) would be lost when merged by time."""
        with self._lock:
            t = max(time.time(), self._clock[0] + 1e-6)
            self._clock[0] = t
            return t

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
                   (run_id, self._ts(), objective, json.dumps(config or {}, default=str), 'running'))
        return self.for_run(run_id)

    def finish_run(self, status='finished', final_report=None):
        self._exec("UPDATE runs SET status=?, final_report=COALESCE(?, final_report) WHERE run_id=?",
                   (status, _text(final_report), self.run_id))

    def request_cancel(self):
        self._exec("UPDATE runs SET status='cancelling' WHERE run_id=? AND status='running'", (self.run_id,))

    def status(self):
        r = self._q("SELECT status FROM runs WHERE run_id=?", (self.run_id,))
        return r[0][0] if r else None

    def is_cancelled(self):
        return self.status() == 'cancelling'

    def runs(self):
        return self._q("SELECT run_id,ts,objective,status FROM runs ORDER BY ts DESC")

    def run_info(self):
        r = self._q("SELECT run_id,ts,objective,config,status,final_report FROM runs WHERE run_id=?", (self.run_id,))
        return r[0] if r else None

    # ---- writes
    def board(self, agent, kind, content):
        return self._exec("INSERT INTO board(run_id,ts,agent,kind,content) VALUES(?,?,?,?,?)",
                          (self.run_id, self._ts(), agent, kind, _text(content)))

    def artifact(self, agent, name, content, kind='finding'):
        return self._exec("INSERT INTO artifacts(run_id,ts,agent,name,kind,content) VALUES(?,?,?,?,?,?)",
                          (self.run_id, self._ts(), agent, _text(name), kind, _text(content)))

    def access(self, reader, artifact_id, action="read"):
        """Log a read and return the artifact row, or None if it doesn't exist in this run."""
        art = self.get_artifact(artifact_id)
        if art is None:
            return None
        author = art[2]
        self._exec("INSERT INTO accesses(run_id,ts,reader,artifact_id,author,cross_agent,action) VALUES(?,?,?,?,?,?,?)",
                   (self.run_id, self._ts(), reader, art[0], author, int(author != reader), action))
        return art

    def trace(self, agent, event, payload):
        return self._exec("INSERT INTO traces(run_id,ts,agent,event,payload) VALUES(?,?,?,?,?)",
                          (self.run_id, self._ts(), agent, event, json.dumps(payload, default=str)))

    def log_rows(self, agent, table, row_ids):
        """Record that `agent` looked at these row ids of `table`."""
        if not row_ids:
            return
        return self._exec("INSERT INTO row_views(run_id,ts,agent,tbl,ranges) VALUES(?,?,?,?,?)",
                          (self.run_id, self._ts(), agent, table, json.dumps(to_ranges(row_ids))))

    # ---- reads
    def coverage(self, table, agent=None):
        """Merged ranges of row ids viewed in `table` by everyone, or by one `agent`."""
        sql, args = "SELECT ranges FROM row_views WHERE run_id IS ? AND tbl=?", [self.run_id, table]
        if agent:
            sql, args = sql + " AND agent=?", args + [agent]
        return merge_ranges(r for (txt,) in self._q(sql, tuple(args)) for r in json.loads(txt))

    def coverage_counts(self):
        """{table: {'all': n_rows_viewed, agent: n_rows_viewed, ...}} for the UI."""
        out = {}
        for tbl, agent, txt in self._q("SELECT tbl,agent,ranges FROM row_views WHERE run_id IS ?", (self.run_id,)):
            out.setdefault(tbl, {}).setdefault(agent, []).extend(json.loads(txt))
        return {t: {**{a: sum(b - a_ + 1 for a_, b in merge_ranges(r)) for a, r in per.items()},
                    'all': sum(b - a_ + 1 for a_, b in merge_ranges([x for r in per.values() for x in r]))}
                for t, per in out.items()}

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
        return self.since({})

    def since(self, last):
        """Rows added after the given per-table id cursors, e.g. since({'traces': 41}). Used for live streaming."""
        return {t: self._q(f"SELECT {cols} FROM {t} WHERE run_id IS ? AND id>? ORDER BY id",
                           (self.run_id, int(last.get(t, 0)))) for t, cols in READ_COLS.items()}

    def coverage_detail(self):
        """{table: {'ranges': merged ranges read by anyone, 'agents': {agent: rows read}}}"""
        raw = {}
        for tbl, agent, txt in self._q("SELECT tbl,agent,ranges FROM row_views WHERE run_id IS ?", (self.run_id,)):
            raw.setdefault(tbl, {}).setdefault(agent, []).extend(json.loads(txt))
        return {t: {'ranges': merge_ranges([x for r in per.values() for x in r]),
                    'agents': {a: sum(b - a_ + 1 for a_, b in merge_ranges(r)) for a, r in per.items()}}
                for t, per in raw.items()}
