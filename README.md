# Decentralized Multi-Agent Shared-Workspace Sandbox V1

Identical agents (default five) pursue one common objective with no orchestrator, no assigned roles, and no direct agent-to-agent messaging. Agents can optionally inspect shared public artifacts, query an uploaded dataset, and publish artifacts to a common workspace.

## Quick start

1. Create a venv: `python -m venv .venv`, then activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere)
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and set `PROVIDER`, `MODEL`, `API_KEY` (and `BASE_URL` if needed). These can also be set in the UI sidebar.
4. `streamlit run ui/app.py`
5. Use **Test connection**, optionally upload a CSV/Excel/Parquet file, enter an objective and run.

Tests (no network or key needed): `python -m pytest -q`

## Model providers

| PROVIDER | Talks to | Key |
|---|---|---|
| `openai` | OpenAI API (or any OpenAI-compatible `BASE_URL`) | required |
| `anthropic` | Anthropic Messages API | required |
| `litellm` | LiteLLM proxy (`http://localhost:4000` by default) | proxy key if configured |
| `local` | Ollama / LM Studio / vLLM / llama.cpp (`http://localhost:11434/v1` by default) | not needed |

## How it works

- Each agent is an independent LangGraph graph whose single `act` node loops until the agent says `done` or hits its step limit. Agents run in parallel threads and start with a small random delay so they don't all act on an identical empty workspace.
- On every step an agent sees: the objective, the dataset profile (if a file was uploaded), the latest shared board posts, an **index** of shared artifacts (titles and short previews), its own previous steps, and the results of what it asked for last step.
- Tools (fields in the agent's JSON reply):
  - `data_query`: a declarative query (filters, derived columns, group-by, aggregates, ratios, sorting, correlation) run with pandas. No model-generated code is executed. Results come back next step and are saved as public `query_result` artifacts.
  - `artifact_to_read`: delivers the full artifact next step and logs the access (flagged as cross-agent when the reader is not the author).
  - `board_message`, `artifact_name` + `artifact_content`: publish to the workspace.
- Uploaded data is lightly cleaned (header and value whitespace, case-only duplicates such as `FRANCE`/`France`, blank rows, Excel serial dates). Every change is listed in the profile the agents see. Remaining issues (duplicates, near-duplicate categories, missing values) are reported, not fixed.
- Every run gets its own `run_id` in `workspace.sqlite`, so runs never contaminate each other. The UI lets you browse past runs.
- Optional **post-hoc report**: after all agents finish, an observer summarises the workspace. Agents never see it, so it doesn't make the swarm orchestrated.

## Important research constraint

Do not store or display raw hidden chain-of-thought. The trace is an audit trace containing observations, actions, artifact references, confidence, and outcomes.
