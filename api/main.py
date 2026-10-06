"""FastAPI service around the swarm.

  GET    /api/config                       defaults from .env (never returns the API key itself)
  POST   /api/llm/test                     check a model configuration
  POST   /api/datasets                     multipart upload of one or more CSV / Excel / Parquet files
  GET    /api/datasets/{id}                table list + profile
  GET    /api/datasets/{id}/rows           paged preview of one table
  DELETE /api/datasets/{id}
  GET    /api/runs                         past runs
  POST   /api/runs                         start a run (returns immediately; agents work in a background thread)
  GET    /api/runs/{id}                    summary: status, counters, agents, row coverage, report
  POST   /api/runs/{id}/stop
  POST   /agui                             AG-UI endpoint (https://docs.ag-ui.com): body = RunAgentInput, response = SSE
                                           stream of AG-UI events. forwardedProps starts a run
                                           ({llm, nAgents, steps, datasetId, report}) or re-attaches to one ({attachRunId}).
Run:  python -m uvicorn api.main:app --port 8000     then open http://127.0.0.1:8000
"""
import asyncio, json, os, threading, uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from ag_ui.core import RunAgentInput, RunErrorEvent, RunStartedEvent
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')

from swarm.agui import AguiTranslator
from swarm.data import DataCollection
from swarm.experiment import launch
from swarm.llm import DEFAULT_BASE_URLS, KEY_ENV, PROVIDERS, LLMConfig, ping
from swarm.summary import ACTIVE, RunView
from swarm.workspace import Workspace

MAX_UPLOAD_MB = 50
MAX_DATASETS = 20
STREAM_TICK = 0.5


class LLMIn(BaseModel):
    provider: str = 'openai'
    model: str = ''
    api_key: str = ''
    base_url: str = ''
    temperature: Optional[float] = 0.7      # null = do not send (for models that reject it)


class RunIn(BaseModel):
    objective: str = Field(min_length=1)
    llm: LLMIn
    n_agents: int = Field(5, ge=1, le=8)
    steps: int = Field(10, ge=1, le=40)
    dataset_id: Optional[str] = None
    report: bool = True


def build_cfg(req: LLMIn) -> LLMConfig:
    """Request values win; blanks fall back to .env so keys can stay out of the browser."""
    env = LLMConfig.from_env()
    provider = (req.provider or env.provider).strip().lower()
    key_env = KEY_ENV.get(provider)
    return LLMConfig(
        provider=provider,
        model=(req.model or (env.model if env.provider == provider else '')).strip(),
        api_key=(req.api_key or os.getenv('API_KEY') or (os.getenv(key_env) if key_env else '') or '').strip(),
        base_url=(req.base_url or (env.base_url if env.provider == provider else '')).strip(),
        temperature=req.temperature, max_tokens=env.max_tokens, timeout=env.timeout)


class _Upload:
    def __init__(self, name, data): self.name, self._d = name, data
    def getvalue(self): return self._d


def create_app(db_path: Optional[str] = None) -> FastAPI:
    db_path = str(db_path or os.getenv('WORKSPACE_DB') or ROOT / 'workspace.sqlite')
    app = FastAPI(title='Decentralized Swarm')
    ws_all = Workspace(db_path)
    datasets: 'OrderedDict[str, dict]' = OrderedDict()
    threads: dict = {}
    lock = threading.Lock()

    def alive(run_id):
        t = threads.get(run_id)
        return bool(t and t.is_alive())

    # runs left 'running' by a dead server process can never finish
    for r in ws_all.runs():
        if r[3] in ACTIVE and not alive(r[0]):
            ws_all.for_run(r[0]).finish_run('interrupted')

    def active_run():
        return next((r[0] for r in ws_all.runs() if r[3] in ACTIVE and alive(r[0])), None)

    def get_run(run_id):
        ws = ws_all.for_run(run_id)
        info = ws.run_info()
        if info is None:
            raise HTTPException(404, f'unknown run {run_id}')
        return ws, info

    def dataset_summary(did, d):
        c = d['collection']
        return {'dataset_id': did, 'files': d['files'], 'profile': c.profile,
                'tables': [{'name': n, 'rows': len(t.df), 'columns': [str(x) for x in t.df.columns],
                            'cleaning': t.notes} for n, t in c.tables.items()]}

    # ---------------- config / model
    @app.get('/api/config')
    def config():
        env = LLMConfig.from_env()
        return {'providers': PROVIDERS, 'default_base_urls': DEFAULT_BASE_URLS,
                'defaults': {'provider': env.provider, 'model': env.model, 'base_url': env.base_url,
                             'temperature': env.temperature},
                'key_in_env': {p: bool(os.getenv(k)) for p, k in KEY_ENV.items() if k} | {'generic': bool(os.getenv('API_KEY'))}}

    @app.post('/api/llm/test')
    def llm_test(req: LLMIn):
        try:
            cfg = build_cfg(req).validate()
            import time
            t = time.time()
            reply = ping(cfg)
            return {'ok': True, 'seconds': round(time.time() - t, 2), 'reply': reply[:200]}
        except Exception as e:
            return JSONResponse({'ok': False, 'error': f'{type(e).__name__}: {e}'}, status_code=200)

    # ---------------- datasets
    @app.post('/api/datasets')
    async def upload(files: list[UploadFile] = File(...)):
        sources, names = [], []
        for f in files:
            data = await f.read()
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f'{f.filename} is larger than {MAX_UPLOAD_MB} MB')
            sources.append((_Upload(f.filename, data), f.filename))
            names.append(f.filename)
        try:
            collection = await asyncio.to_thread(DataCollection.from_sources, sources)
        except Exception as e:
            raise HTTPException(400, f'Could not read files: {type(e).__name__}: {e}')
        did = uuid.uuid4().hex[:10]
        with lock:
            datasets[did] = {'collection': collection, 'files': names}
            while len(datasets) > MAX_DATASETS:
                datasets.popitem(last=False)
        return dataset_summary(did, datasets[did])

    def get_dataset(did):
        d = datasets.get(did)
        if d is None:
            raise HTTPException(404, 'unknown dataset (uploads live in server memory; upload again after a restart)')
        return d

    @app.get('/api/datasets/{did}')
    def dataset_info(did: str):
        return dataset_summary(did, get_dataset(did))

    @app.get('/api/datasets/{did}/rows')
    def dataset_rows(did: str, table: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
        c = get_dataset(did)['collection']
        try:
            df = c._get(table).df
        except ValueError as e:
            raise HTTPException(404, str(e))
        page = df.iloc[offset:offset + limit].copy()
        for col in page.columns:
            if str(page[col].dtype).startswith('datetime'):
                page[col] = page[col].dt.strftime('%Y-%m-%d')
        out = json.loads(page.to_json(orient='split', index=True))
        return {'table': table, 'total': len(df), 'offset': offset, 'columns': out['columns'],
                'row_ids': out['index'], 'rows': out['data']}

    @app.delete('/api/datasets/{did}')
    def dataset_delete(did: str):
        with lock:
            datasets.pop(did, None)
        return {'ok': True}

    # ---------------- runs
    @app.get('/api/runs')
    def runs():
        return [{'run_id': r[0], 'started': r[1], 'objective': r[2], 'status': r[3], 'active': r[3] in ACTIVE}
                for r in ws_all.runs()]

    def begin_run(req: RunIn) -> str:
        """Validate, then start a swarm run in a background thread. Raises HTTPException(400/404/409)."""
        try:
            cfg = build_cfg(req.llm).validate()
        except ValueError as e:
            raise HTTPException(400, str(e))
        collection = get_dataset(req.dataset_id)['collection'] if req.dataset_id else None
        with lock:
            busy = active_run()
            if busy:
                raise HTTPException(409, f'run {busy} is still active; stop it or wait for it to finish')
            run_id, thread = launch(req.objective, cfg, req.n_agents, req.steps, db_path, collection, do_synthesis=req.report)
            threads[run_id] = thread
        return run_id

    @app.post('/api/runs', status_code=201)
    def start_run(req: RunIn):
        return {'run_id': begin_run(req)}

    @app.get('/api/runs/{run_id}')
    def run_summary(run_id: str):
        ws, info = get_run(run_id)
        view = RunView()
        view.update(ws)
        return view.status(ws, ws.run_info())

    @app.post('/api/runs/{run_id}/stop')
    def stop(run_id: str):
        ws, info = get_run(run_id)
        if info[4] not in ACTIVE:
            raise HTTPException(409, f'run is {info[4]}, nothing to stop')
        ws.request_cancel()
        return {'ok': True, 'status': 'cancelling'}

    # ---------------- AG-UI
    @app.post('/agui')
    async def agui(request: Request):
        try:
            inp = RunAgentInput.model_validate(await request.json())
        except Exception as e:
            raise HTTPException(422, f'not a valid AG-UI RunAgentInput: {e}')
        encoder = EventEncoder(accept=request.headers.get('accept'))
        fp = inp.forwarded_props if isinstance(inp.forwarded_props, dict) else {}

        def failed(msg, code):
            async def one():
                yield encoder.encode(RunStartedEvent(thread_id=inp.thread_id, run_id=inp.run_id))
                yield encoder.encode(RunErrorEvent(message=msg, code=code))
            return StreamingResponse(one(), media_type=encoder.get_content_type())

        try:
            if fp.get('attachRunId'):                         # re-open a past or still-running run
                swarm_run = fp['attachRunId']
                await asyncio.to_thread(get_run, swarm_run)
            else:                                             # start a new run from the user's message + forwardedProps
                text = next((m.content for m in reversed(inp.messages) if m.role == 'user' and isinstance(getattr(m, 'content', None), str)), '')
                req = RunIn(objective=text or '', llm=LLMIn(**(fp.get('llm') or {})), n_agents=fp.get('nAgents', 5),
                            steps=fp.get('steps', 10), dataset_id=fp.get('datasetId'), report=fp.get('report', True))
                swarm_run = await asyncio.to_thread(begin_run, req)
        except HTTPException as e:
            return failed(str(e.detail), f'http_{e.status_code}')
        except Exception as e:
            return failed(f'{type(e).__name__}: {e}', 'bad_request')

        ws = ws_all.for_run(swarm_run)
        tr = AguiTranslator(ws, inp.thread_id, inp.run_id)

        async def gen():
            for ev in await asyncio.to_thread(tr.begin):
                yield encoder.encode(ev)
            while not tr.done:
                for ev in await asyncio.to_thread(tr.poll):
                    yield encoder.encode(ev)
                if not tr.done:
                    await asyncio.sleep(STREAM_TICK)    # a dropped connection only ends this generator; the swarm run goes on

        return StreamingResponse(gen(), media_type=encoder.get_content_type(),
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    # ---------------- React frontend (frontend/dist, built with `npm run build`)
    dist = ROOT / 'frontend' / 'dist'
    if dist.is_dir():
        app.mount('/', StaticFiles(directory=dist, html=True), name='frontend')
    else:
        @app.get('/')
        def no_frontend():
            return JSONResponse({'message': 'API is running. Build the UI with: cd frontend && npm install && npm run build '
                                            '(or run `npm run dev` there for hot reload).', 'docs': '/docs'})

    app.state.workspace = ws_all
    return app


_app = None


def __getattr__(name):
    """`uvicorn api.main:app` resolves this lazily, so importing the module (e.g. in tests) never opens the real database."""
    global _app
    if name == 'app':
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
