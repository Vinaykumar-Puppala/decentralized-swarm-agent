# Decentralized Multi-Agent Shared-Workspace Sandbox

Identical agents (default five) pursue one common objective with no orchestrator, no assigned roles, and no direct agent-to-agent messaging. Agents can optionally inspect shared public artifacts, query and read the rows of uploaded data, and publish artifacts to a common workspace.

```
React UI (frontend/)  <--- AG-UI events over SSE --->  FastAPI (api/)  --->  swarm/  (the autonomous agents, unchanged)
 @ag-ui/client HttpAgent        POST /agui              AguiTranslator        LangGraph agents + SQLite workspace
```

The agents know nothing about the web layer. FastAPI starts a run in a background thread and an `AguiTranslator` turns what the agents write to the shared workspace into [AG-UI](https://docs.ag-ui.com) events. A browser tab closing or reloading never stops a run; the page simply re-attaches.

## Quick start

1. Python env: `python -m venv .venv`, activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere), then `pip install -r requirements.txt`
2. Model settings (optional, can also be entered in the UI): `cp .env.example .env` and set `PROVIDER`, `MODEL`, `API_KEY` (and `BASE_URL` if needed)
3. Build the UI once (Node 18+): `cd frontend && npm install && npm run build`
4. Start the server: `python -m uvicorn api.main:app --port 8000`, then open <http://127.0.0.1:8000>
5. **Test connection**, optionally drop in several CSV / Excel / Parquet files, write the objective, run.

Frontend development with hot reload: run the API as above, then `cd frontend && npm run dev` and open <http://localhost:5173> (it proxies `/api` and `/agui` to port 8000).

Tests (no network or API key needed): `python -m pytest -q`

## Model providers

| PROVIDER | Talks to | Key |
|---|---|---|
| `openai` | OpenAI API (or any OpenAI-compatible `BASE_URL`) | required |
| `anthropic` | Anthropic Messages API | required |
| `litellm` | LiteLLM proxy (`http://localhost:4000` by default) | proxy key if configured |
| `local` | Ollama / LM Studio / vLLM / llama.cpp (`http://localhost:11434/v1` by default) | not needed |

An API key typed in the UI is sent to the server for that run only. It is not stored in the browser or in the run log; leave it blank to use the key from the server's `.env`.

## AG-UI mapping

`POST /agui` takes an AG-UI `RunAgentInput` and answers with a stream of AG-UI events. `forwardedProps` carries `{llm, nAgents, steps, datasetId, report}` to start a run, or `{attachRunId}` to replay and follow an existing one.

| Swarm | AG-UI |
|---|---|
| one experiment | one run: `RUN_STARTED` ... `RUN_FINISHED` (`outcome` success / cancelled) or `RUN_ERROR` |
| each autonomous agent (and the reporter) | a sub-agent: `SUBAGENT_STARTED` / `FINISHED` / `ERROR`, and every event it causes carries its `subagentRunId` |
| an agent step | `STEP_STARTED` / `STEP_FINISHED` (`step-1`, `step-2`, ...) |
| audit summary of a decision | `TEXT_MESSAGE_*` (role assistant, `name` = agent) |
| `data_query`, `data_rows`, `read_artifact` | `TOOL_CALL_START` / `ARGS` / `END`, then `TOOL_CALL_RESULT` |
| board post, artifact | `ACTIVITY_SNAPSHOT` (`board_post`, `artifact`) |
| artifact reads, agent errors, stop | `CUSTOM` (`swarm.access`, `swarm.agent_error`, `swarm.agent_cancelled`) |
| counters, per-agent status, row coverage | `STATE_SNAPSHOT`, then `STATE_DELTA` (JSON Patch) |
| post-hoc report | `TEXT_MESSAGE_*` streamed in chunks (`name` = reporter) |

Other endpoints: `GET /api/config`, `POST /api/llm/test`, `POST /api/datasets` (multipart, several files), `GET /api/datasets/{id}/rows`, `GET /api/runs`, `GET /api/runs/{id}`, `POST /api/runs/{id}/stop`. Interactive docs at `/docs`. Only one run is active at a time.

## How the agents work

- Each agent is an independent LangGraph graph whose single `act` node loops until the agent says `done` or hits its step limit. Agents run in parallel threads and start with a small random delay so they don't all act on an identical empty workspace.
- On every step an agent sees: the objective, the dataset profile (if files were uploaded), the latest shared board posts, an **index** of shared artifacts (titles and short previews), its own previous steps, and the results of what it asked for last step.
- Tools (fields in the agent's JSON reply):
  - `data_query`: a declarative query (filters, derived columns, group-by, aggregates, ratios, sorting, correlation) run with pandas. No model-generated code is executed. Results come back next step and are saved as public `query_result` artifacts.
  - `data_rows`: reads the actual rows of a table, up to 100 per call, with optional filters, sorting and column selection. Every row has a stable `_row` id. The workspace tracks which row ids have been read, and each agent's prompt shows the swarm-wide coverage and the ranges still unread, so agents spread out on their own and, given enough steps, cover every row. Coverage is advisory: nothing assigns rows to agents.
  - `artifact_to_read`: delivers the full artifact next step and logs the access (flagged as cross-agent when the reader is not the author).
  - `board_message`, `artifact_name` + `artifact_content`: publish to the workspace.
- **Activity log and scratchpads.** Every step is recorded in a shared activity log: what the agent tried, a detailed written `rationale` (the explicit reason for the step; never hidden chain-of-thought), each tool with its arguments and outcome (failures included), and a confidence. Each agent also has a private scratchpad for its own notes, which come back in its own prompts. Peers see **pointers only** ("the log has N entries", "agent-2 has 3 notes"); nothing is shown unless an agent decides to ask with `read_log` or `read_scratchpad`, and reading is never required. Each such read is itself written to the log as a `read` entry, so you can see who looked at whose work. The rationale also appears in the live feed.
- **Visibility** (`visibility` in the API, default `full`) controls how much an agent may choose to read: `isolated` (no peer reading; nothing advertised), `log` (the activity log), `full` (log and other agents' scratchpads). Use it to compare how much sharing changes what the swarm finds.
- Upload several files at once: each CSV/Parquet file and each Excel sheet becomes its own table. Tables are never joined; agents set `"table"` in `data_query` / `data_rows` to choose one.
- Uploaded data is lightly cleaned (header and value whitespace, case-only duplicates such as `FRANCE`/`France`, blank rows, Excel serial dates). Every change is listed in the profile the agents see. Remaining issues (duplicates, near-duplicate categories, missing values) are reported, not fixed.
- Every run gets its own `run_id` in `workspace.sqlite`, so runs never contaminate each other. Timestamps are strictly increasing, so the order of events is always causal.
- Optional **post-hoc report**: after all agents finish, an observer summarises the workspace. Agents never see it, so it doesn't make the swarm orchestrated.

`ui/app.py` is the earlier Streamlit front end. It still works (`streamlit run ui/app.py`) but is no longer the main UI.

## Important research constraint

Do not store or display raw hidden chain-of-thought. The trace is an audit trace containing observations, actions, artifact references, confidence, and outcomes. The UI labels agent messages accordingly.
