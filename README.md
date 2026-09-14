# orchestration-agent

Decoupled, opt-in orchestration agent implementing `AGENT_ORCHESTRATION_SPEC.md`:
skill loading from `SKILL.md` files, an OpenAI provider, external tool execution,
token tracking, retries, rate limiting, and pluggable conversation history
(in-memory + MongoDB).

See the plan/spec for full design rationale and the list of intentional
corrections made to the spec's pseudocode when it didn't match the real OpenAI
API (`src/orchestration_agent/provider/openai_provider.py` and
`src/orchestration_agent/agent.py` docstrings).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[openai,mongodb,redis,dev]"
docker compose up -d   # mongodb + redis, for integration tests
```

`.env` must define `OPENAI_API_KEY` (see `.env.example` for the full list of
variables used by tests/examples).

## Run

```bash
.venv/bin/pytest -q                 # unit + live OpenAI + docker-backed tests
.venv/bin/python examples/basic_usage.py   # end-to-end demo against gpt-5.4-mini
docker compose down
```

## Layout

- `src/orchestration_agent/` — the library (agent loop, provider, skills, tools,
  conversation history, rate limiting, logging).
- `skills/` — example `SKILL.md` files (`code-review`, `sql-analysis`).
- `examples/basic_usage.py` — real end-to-end run against `gpt-5.4-mini`.
- `tests/` — unit tests (no network) plus integration tests gated on Docker
  services / `OPENAI_API_KEY` being available.
