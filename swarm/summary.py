"""Turns raw workspace rows into UI-friendly events and a running summary (agent states, counters, coverage).
Pure Python, no web framework: the FastAPI layer and the tests both use it."""
import json, time

ACTIVE = ('running', 'cancelling')
ERROR_EVENTS = ('error', 'query_error', 'run_failed')
DONE_STATES = ('finished', 'failed', 'stopped')


def _payload(txt):
    try:
        return json.loads(txt or '{}')
    except (TypeError, ValueError):
        return {'raw': txt}


class RunView:
    """Incrementally consumes new rows of one run. `update(ws)` returns the new events (as dicts);
    `status(ws, info)` returns the current summary. Keeps O(new rows) work per call."""

    def __init__(self):
        self.cursor = {}
        self.last_ts = None
        self.agents = {}
        self.counts = {'board': 0, 'findings': 0, 'finals': 0, 'queries': 0, 'row_reads': 0,
                       'cross_reads': 0, 'errors': 0, 'events': 0}

    # ---- consuming rows
    def update(self, ws):
        new = ws.since(self.cursor)
        events = []
        for i, ts, agent, event, payload in new['traces']:
            p = _payload(payload)
            self._track(ts, agent, event, p)
            events.append({'type': 'trace', 'id': i, 'ts': ts, 'agent': agent, 'event': event, 'payload': p})
        for i, ts, agent, kind, content in new['board']:
            self.counts['board'] += 1
            events.append({'type': 'board', 'id': i, 'ts': ts, 'agent': agent, 'kind': kind, 'content': content})
        for i, ts, agent, name, kind, content in new['artifacts']:
            key = {'finding': 'findings', 'final_answer': 'finals', 'query_result': 'queries'}.get(kind)
            if key:
                self.counts[key] += 1
            events.append({'type': 'artifact', 'id': i, 'ts': ts, 'agent': agent, 'name': name, 'kind': kind, 'content': content})
        for i, ts, reader, aid, author, cross, action in new['accesses']:
            self.counts['cross_reads'] += int(bool(cross))
            events.append({'type': 'access', 'id': i, 'ts': ts, 'reader': reader, 'artifact_id': aid, 'author': author,
                           'cross_agent': bool(cross), 'action': action})
        for table, rows in new.items():
            if rows:
                self.cursor[table] = rows[-1][0]
        self.counts['events'] += len(new['traces'])
        events.sort(key=lambda e: (e['ts'], e['id']))
        if events:
            self.last_ts = events[-1]['ts']
        return events

    def _track(self, ts, agent, event, p):
        a = self.agents.setdefault(agent, {'steps': 0, 'last_ts': ts, 'action': '', 'summary': '', 'state': 'working', 'errors': 0})
        a['last_ts'] = ts
        if event == 'step_started':
            a['state'] = 'writing report' if agent == 'reporter' else f"waiting on model (step {p.get('step')})"
        elif event == 'decision':
            a.update(steps=a['steps'] + 1, action=p.get('action') or '', summary=p.get('audit_summary') or '', state='working')
        elif event in ERROR_EVENTS:
            a['errors'] += 1
            self.counts['errors'] += 1
        elif event == 'rows_viewed':
            self.counts['row_reads'] += 1
        elif event == 'run_finished':
            a['state'] = 'finished'
        elif event == 'run_failed':
            a['state'] = 'failed'
        elif event == 'cancelled':
            a['state'] = 'stopped'

    # ---- summary
    def status(self, ws, info, now=None):
        """info: the runs-table row (run_id, ts, objective, config, status, final_report)."""
        now = now or time.time()
        run_id, ts, objective, config, status, report = info
        config = _payload(config)
        sizes = config.get('tables') or {}
        cov = ws.coverage_detail()
        coverage = {}
        for t, total in sizes.items():
            c = cov.get(t, {'ranges': [], 'agents': {}})
            coverage[t] = {'total': total, 'read': min(total, sum(b - a + 1 for a, b in c['ranges'])),
                           'ranges': c['ranges'], 'agents': c['agents']}
        agents = []
        for name, a in sorted(self.agents.items(), key=lambda kv: (kv[0] == 'reporter', kv[0])):
            agents.append({'agent': name, 'state': a['state'], 'steps': a['steps'], 'errors': a['errors'],
                           'rows_read': sum(t['agents'].get(name, 0) for t in coverage.values()),
                           'idle_s': None if a['state'] in DONE_STATES else round(now - a['last_ts']),
                           'action': a['action'], 'summary': a['summary']})
        return {'run_id': run_id, 'status': status, 'active': status in ACTIVE, 'started': ts, 'elapsed': round((now if status in ACTIVE else (self.last_ts or ts)) - ts, 1),
                'objective': objective, 'config': config, 'counts': dict(self.counts), 'agents': agents,
                'coverage': coverage, 'report': report}
