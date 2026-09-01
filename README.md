# Local Coding Agent

This repository provides the building blocks for a multi-specialist coding agent. A caller can select one of four agent modes—backend, ML/AI, Git, or algorithms—or let `AgentRunner` route a request automatically. It then invokes the matching LangGraph workflow with a task-specific prompt, model configuration, and allow-listed tools.

The repository is primarily a library, with `main.py` also providing an interactive terminal runner.

## Architecture

```text
Caller / future Streamlit UI
          |
          v
  Config.from_env() -----> Config (providers, models, safety, limits)
          |                           |
          v                           v
     AgentRunner -----> selected BaseAgent subclass
          |                    (backend / ml / git / algorithms)
          |                           |
          |                           v
          |                 LangGraph state machine
          |                 agent -> confirmation -> tools --+
          |                   ^                                |
          |                   +--------------------------------+
          |                           |
          v                           v
    AgentRunResult              LLM provider chain + task tools
                         SambaNova -> OpenRouter -> Groq -> Ollama

Optional integrations enabled from the Streamlit sidebar:
  database/  -> SQLAlchemy persistence for sessions, messages, tools, and approvals
  sandbox/   -> hardened Docker service for isolated Python/shell execution
```

### Agent execution flow

1. The caller selects an agent key (or passes `None` for automatic routing) and sends a message to `AgentRunner.run()`.
2. `AgentRunner` lazily creates the selected agent and gives it only that task's tool set.
3. `BaseAgent` adds the task's Markdown system prompt, invokes a tool-bound LLM, and records the response in `AgentState`.
4. If the model requests tools, the graph checks each call for confirmation. Calls requiring approval pause through LangGraph `interrupt()`.
5. On approval (or when no approval is required), tools run and their outputs are appended to state. The graph returns to the model until it produces a final response, fails, or reaches the iteration limit.
6. The runner returns the final LangGraph state together with the selected agent key and thread ID. A paused run continues through `resume_confirmation()` using the same thread ID.

The compiled graphs use an in-memory LangGraph checkpointer. Keeping one `AgentRunner` instance alive (for example in Streamlit session state) is therefore necessary to preserve paused confirmations and conversation state. Passing `project_path` pins tool paths to that project; `enable_worktree=True` creates an isolated scratch worktree for each conversation.

## Project layout

| Path | Responsibility | Connected to |
| --- | --- | --- |
| `main.py` | Placeholder executable that prints a greeting. | Not yet connected to the agent runtime. |
| `config.py` | Pydantic configuration: API keys, provider URLs, per-task models, retry/context limits, confirmation policy, PostgreSQL, and Celery settings. `Config.from_env()` loads required provider keys. | Passed to agents, model clients, and persistence queueing. |
| `agent/__init__.py` | Package entry point and frontend-facing orchestration API. Exports `AgentRunner`, which lists selectable agents, constructs them lazily, starts runs, and resumes confirmation interrupts. | Uses agent graphs and `TOOLS_BY_TASK`. |
| `agent/graphs/base_graph.py` | Shared LangGraph implementation: prompt loading, LLM calls, retries, context recovery, confirmation gates, tool invocation, logs, routing, and in-memory checkpointing. | Base class for every specialist graph. |
| `agent/graphs/backend_graph.py` | Backend specialization (`TASK_MODE = "backend"`). | Inherits the base workflow and backend tools/prompt. |
| `agent/graphs/ml_graph.py` | ML/AI specialization (`TASK_MODE = "ml"`). | Inherits the base workflow and ML tools/prompt. |
| `agent/graphs/git_graph.py` | Git specialization (`TASK_MODE = "git"`). | Inherits the base workflow and Git tools/prompt. |
| `agent/graphs/algo_graph.py` | Algorithms specialization (`TASK_MODE = "algorithms"`), including its `algo_agent.md` prompt override. | Inherits the base workflow and algorithm tools/prompt. |
| `agent/graphs/md/*.md` | System prompts that constrain each specialist's working style. | Loaded by `BaseAgent` from the agent's task mode. |
| `agent/graphs/helper.py` | Graph support functions: error classification, retry timing, thread ID lookup, leaked tool-call repair, output trimming, and read-only-loop detection. | Called by `BaseAgent`. |
| `agent/task_profile.py` | Single source of truth mapping task modes to tool names, descriptions, and config model fields. | Filters tools in `BaseAgent`; resolves task model in `Config`. |
| `agent/llm.py` | Builds cached LangChain chat clients and fallback chains. API mode is SambaNova → OpenRouter → Groq → optional Ollama; local mode uses Ollama only. | Tool-bound model is used by `BaseAgent`. |
| `agent/context_window.py` | Keeps recent conversation turns and summarizes older history with a local Ollama summary model when the context budget is exceeded. | Used when an LLM call reports a context-length error. |
| `agent/confirmation.py` | Identifies dangerous operations, converts confirmation requests/decisions to JSON-safe payloads, and includes Streamlit/CLI confirmation helpers. | `BaseAgent` pauses with these payloads; `AgentRunner` resumes them. |
| `agent/storage.py` | Best-effort adapter that records graph messages, executed tool calls, and confirmation decisions through the database event service. | Optional dependency injected into `AgentRunner`. |
| `agent/tools/` | LangChain tools for filesystem, shell, search, HTTP/OpenAPI, Git, Jupyter kernels, sandbox execution, and system/background-process operations. | Grouped by task in `agent/tools/__init__.py`. |
| `schema/agent_schema.py` | Runtime Pydantic schemas: LangGraph `AgentState`, confirmation payloads, tool logs, and serializable agent results. | Used by graphs and confirmation logic. |
| `schema/database_schema.py` | Pydantic input/output schemas for persisted sessions, messages, tool calls, sandbox runs, confirmations, summaries, memory, agent files, and usage. | Validates database persistence events. |
| `schema/base.py` | Common Pydantic schema configuration. | Base class for project schemas. |
| `log/log_event.py` | Thread-safe, best-effort JSON-lines event logger. | Used by the base graph for model, retry, confirmation, and tool events. |
| `database/base.py` | SQLAlchemy declarative base, enums, and UTC timestamp helper. | Shared by database models and schemas. |
| `database/session.py` | SQLAlchemy engine/session lifecycle, schema initialization, transactions, health checks, and singleton manager. | Used by `DatabaseLogService`. |
| `database/session_manager.py` | ORM `Session` model and its relationships. | Parent for persisted agent data. |
| `database/backend_agent.py` | ORM models for messages, tool calls, sandbox runs, confirmations, summaries, memory facts, agent-file versions, and provider usage. | Related to `Session`; written by persistence service. |
| `database/service/persistence.py` | Validates named persistence events and performs database creates/updates. | Uses database schemas and ORM models. |
| `database/service/celery_app.py` | Creates/configures the Celery application. | Used by asynchronous persistence tasks. |
| `database/service/tasks.py` | Celery task for event persistence and helper to enqueue it through Redis. | Calls `DatabaseLogService`; accepts `Config`. |
| `sandbox/sandbox_config.py` | Docker sandbox resource, filesystem, network, and security-hardening configuration. | Used by `DockerSandbox`. |
| `sandbox/sandbox_manager.py` | Builds, starts, health-checks, calls, and removes the isolated Docker container. | Invoked by the optional `execute_in_sandbox` agent tool. |
| `sandbox/server.py` | FastAPI process inside the sandbox with `/health`, `/upload`, `/exec`, and `/reset` endpoints. | Runs Python/shell code under `/workspace`. |
| `sandbox/Dockerfile`, `sandbox/run.sh`, `sandbox/stop.sh`, `sandbox/requirements.txt` | Container image and helper scripts for the sandbox service. | Used when building/running the sandbox. |
| `pyproject.toml` / `uv.lock` | Python project metadata and locked dependency graph. | Defines the runtime dependencies. |

## Agent modes and tools

| Agent key | Purpose | Tools available |
| --- | --- | --- |
| `backend` | Flask/FastAPI routes, business logic, integrations, and tests. | Shell; files; directory/search/web; HTTP and OpenAPI inspection; deletion. |
| `ml` | Training, evaluation, data pipelines, and experiments. | Files; directory/search/web; persistent Jupyter execution/restart; GPU and background-job controls; deletion. |
| `git` | Repository inspection and version-control changes. | Status, diff, log, branch, checkout, commit, and push. |
| `algorithms` | Correctness- and complexity-sensitive implementation work. | Same tool family as backend. |

`agent/tools/__init__.py` is the concrete registry. `agent/task_profile.py` filters that registry again by tool name, so the graph cannot expose a tool outside the selected profile.

## Safety and resilience

- The agent always requires confirmation for shell commands, Git pushes, path deletion, and launching background processes. Overwriting a file also requires confirmation. Set `AGENT_CONFIRM_ALL_TOOLS=true` to approve every tool call explicitly.
- Confirmation is non-blocking at the graph level: LangGraph interrupts and returns a request payload instead of calling `input()` inside the agent loop.
- Model calls have retry handling for transient errors, rate limits, and local model cold starts. The agent also repairs certain tool calls emitted as plain JSON text.
- Tool output is bounded before it is returned to the model, older tool results are compressed, and context-overflow recovery attempts a local summary before returning an error.
- Runtime events are written as JSON lines to `agent_events.log` by default. Logging is best-effort and never interrupts agent execution.
- The optional Docker sandbox disables network access, uses a read-only root filesystem plus temporary writable paths, drops Linux capabilities, limits memory/CPU/processes, and applies execution timeouts.

## Optional database and sandbox integrations

The direct runtime path is `Config` → `AgentRunner` → selected graph → LLM/tools/confirmation/logging. The Streamlit sidebar can enable two additional integrations for new conversations:

- **Database storage:** creates a database session and records user/assistant/tool messages, completed tool calls, and confirmation decisions. It uses `POSTGRES_URL` (or `Config.postgres_url`) and is best-effort: a storage outage is shown in the UI but does not stop the agent.
- **Docker sandbox execution:** adds `execute_in_sandbox` to the backend, ML, and algorithms tool profiles. Each call starts a temporary hardened Docker container and executes self-contained Python or shell code with no network access, no host-project mount, and resource/time limits. The Git agent intentionally has no code-execution tool.

Changing either toggle opens a separate in-memory conversation because LangGraph checkpoints cannot safely move between tool configurations.

## Minimal integration sketch

```python
from config import Config
from agent import AgentRunner
from agent.storage import AgentStorage
from database.service.persistence import DatabaseLogService

config = Config.from_env()
database = DatabaseLogService(config.postgres_url, echo=config.postgres_echo)
database.init_db()
runner = AgentRunner(
    config,
    enable_sandbox=True,
    storage=AgentStorage(database),
)

result = runner.run(
    agent_key="backend",
    user_message="Inspect the API routes and add a health-check test.",
)

print(result.thread_id)
print(result.state["messages"][-1].content)
```

If this pauses for confirmation, send a decision back with the same selected agent and thread ID:

```python
result = runner.resume_confirmation(
    agent_key="backend",
    thread_id=result.thread_id,
    decision={"approved": True, "reason": "Approved from the UI"},
)
```

## Configuration

`Config.from_env()` requires these environment variables:

- `SAMBANOVA_API_KEY`
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`

Useful optional settings include `AGENT_TYPE`, `AGENT_CONFIRM_ALL_TOOLS`, `AGENT_ENABLE_OLLAMA_FALLBACK`, provider base URLs, and the per-task `BACKEND_MODEL_NAME`, `ML_MODEL_NAME`, `GIT_MODEL_NAME`, and `ALGO_MODEL_NAME` values. Set `provider` to `local` when constructing `Config` to use Ollama only; otherwise API mode uses the configured fallback chain.

For database storage, set `POSTGRES_URL` (plus optional `POSTGRES_POOL_SIZE` and `POSTGRES_ECHO`). The sandbox also requires a running Docker daemon; the agent builds the image from `sandbox/Dockerfile` on first use.

## Implementation Status

### Core Functionality
- ✅ **CLI Entry Point** — Interactive agent runner with rich terminal UI
- ✅ **Agent Router/Dispatcher** — Automatic task classification (backend/ml/git/algorithms)
- ✅ **Safety Tests** — Comprehensive validation of tool filtering and confirmation gates
- ✅ **Git Worktree Isolation** — Optional scratch branches, with review, merge, and discard APIs
- ✅ **Eval Harness** — Local-model benchmark with per-case accuracy and latency results
- ✅ **Context Window Management** — Rolling summaries with local model compression

### Local Ollama Optimization
- ✅ **Ollama Tuning (Phase 5)** — Request settings plus `run_ollama.sh --serve` for daemon flash-attention/KV-cache settings
- ✅ **Tool-Calling Model Default (Phase 6)** — MFDoom/deepseek-coder-v2-tool-calling:16b enforced in local mode
- ✅ **Prompt Tightening (Phase 7)** — Direct tool-call instructions in agent prompts (no narration)
- ✅ **K-Quant Upgrade (Phase 8)** — Support for deepseek-coder-v2:16b-lite-instruct-q4_K_M with benchmarking

### Codebase Retrieval
- ✅ **Phase 10: Local Codebase RAG** — Safe, dependency-free BM25 retrieval with line-cited chunks, automatic freshness checks, and a `search_codebase` tool available to every specialist. Index data is stored outside the repository in the system temporary directory.

### Testing & Validation
- ✅ **Router Tests** — 7 tests validating model defaults and task routing
- ✅ **K-Quant Tests** — 8 tests for quantization mode switching
- ✅ **Safety Test Suite** — 19+ tests for tool filtering, path containment, confirmation gates
- ✅ **Worktree Tests** — Isolation, diff summaries, merge/discard workflows

### Run & Benchmark Scripts
- ✅ `run.sh` — Multi-mode launcher (CLI, full stack, dev, sandbox, celery)
- ✅ `run_ollama.sh` — Ollama readiness check and model configuration display
- ✅ `phase8_validate.sh` — A/B benchmarking script for K-quant vs. tool-calling

### Quick Start
```bash
# Install and check Ollama
bash run_ollama.sh

# Run CLI with automatic agent selection
python3 main.py

# Run with specific agent (backend, ml, git, or algorithms)
python3 main.py --agent backend

# Enable K-quant quantization
OLLAMA_USE_KQUANT=true python3 main.py

# Run a conversation in an isolated Git worktree
python3 main.py --worktree --project /path/to/repo

# Run full stack with sandbox and persistence
bash run.sh full
```

### Local-First Architecture
- All models run locally on Ollama (no cloud API calls by default)
- Fallback chain still supports SambaNova → OpenRouter → Groq → Ollama if configured with API keys
- Tool-calling model with explicit direct-invocation prompts for structured JSON output
- K-quant quantization option for improved accuracy with zero latency overhead
- Optional PostgreSQL persistence and Docker sandbox for isolated execution
