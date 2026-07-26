# Local Python Coding Agent — Requirements & Tools

Personal-use coding agent for Python backend development (Flask/FastAPI) and
ML/DL/data analysis work. Runs locally, single user, no enterprise
orchestration needed.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────┐
│                  LLM (groq)                │
│         via API, function calling         # Local Python Coding Agent — Requirements & Tools

Personal-use coding agent for Python backend development (Flask/FastAPI) and
ML/DL/data analysis work. Runs locally, single user, no enterprise
orchestration needed.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────┐
│                  LLM (groq)                │
│         via API, function calling            │
└───────────────────┬───────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │      Agent Harness       │  (you build this)
        │  - tool routing          │
        │  - context/memory mgmt   │
        │  - checkpointing         │
        └────────────┬────────────┘
                     │
   ┌─────────┬───────┼────────┬──────────┐
   │         │       │        │          │
 Shell     File I/O  Git   Python     Web Search
 Tool       Tool     Tool  Kernel      Tool
```

Core principle: keep the harness thin. Most "tools" should just be shell
commands the LLM invokes, not bespoke wrappers. The two exceptions worth
building custom tools for are **file editing** (patch-based) and a
**persistent Python execution environment** (for ML/data work).

---

## 2. Core Tools (build these first)

### 2.1 Shell/Command Execution Tool
- Executes arbitrary bash commands, returns stdout/stderr/exit code
- Runs inside a project-specific venv or container (see §6)
- Timeout + output truncation (long-running training jobs should be
  backgrounded, not block the tool call — see §2.5)

### 2.2 File Operations Tool
- `read_file(path, line_range=None)` — numbered line output for easy reference
- `write_file(path, content)` — create new files
- `edit_file(path, old_str, new_str)` — targeted patch, must match uniquely
- `list_dir(path, depth=2)` — directory tree, respecting `.gitignore`

Patch-based editing (not full-file rewrite) is important: saves tokens,
avoids clobbering unrelated code, and produces reviewable diffs.

### 2.3 Git Tool
- `git status/diff/log/branch/commit/checkout`
- Auto-commit checkpoint before agent makes changes, so every session is
  reversible with `git reset` or `git revert`
- Consider a dedicated branch per agent session (e.g. `agent/task-name`)
  so you can review before merging to `main`

### 2.4 Code Search Tool
- **ripgrep (`rg`)** for fast text/regex search across the repo
- Optional: tree-sitter or a language server (Pyright) for symbol-level
  search (go-to-definition, find-references) — much better than regex once
  your codebase grows past a few thousand lines

### 2.5 Persistent Python Execution Tool (most important custom tool)
For ML/data work, one-shot shell calls are wasteful (re-importing torch,
reloading a 2GB CSV every call). Build a tool backed by a **persistent
Jupyter kernel**:

- Use `jupyter_client` to start/manage a kernel per project
- `execute_code(code)` → returns stdout, stderr, rich outputs (dataframes,
  matplotlib images as base64, errors with traceback)
- Kernel state persists across calls within a session — variables, loaded
  models, dataframes all stay in memory
- For long-running training: launch as a **background process** via shell
  (`nohup`, `tmux`, or `subprocess.Popen`) instead of blocking the kernel,
  and give the agent a `tail_log(path, n_lines)` tool to check progress

### 2.6 Web Search / Doc Fetch Tool
- For current library docs, error message lookup, API references
- Avoids the agent guessing from stale training data on fast-moving
  libraries (transformers, langchain, fastapi, etc.)

---

## 3. Language & Package Tooling

| Purpose | Tool | Notes |
|---|---|---|
| Dependency mgmt | `uv` (preferred) or `poetry` | `uv` is much faster for repeated agent-driven installs |
| Env isolation | `venv` per project | Never let the agent install globally |
| Multiple Python versions | `pyenv` | If projects pin different versions |
| Linting | `ruff` | Fast, replaces flake8+isort+more |
| Formatting | `black` (or `ruff format`) | Run automatically after edits |
| Type checking | `mypy` or `pyright` | Pyright also gives you LSP-style navigation |
| Testing | `pytest` | Standard; agent should run this after every change |

---

## 4. Flask/FastAPI-Specific Tools

- **uvicorn** (FastAPI) / **flask run** — spin up the app for live testing
- **httpx / curl / httpie** — hit endpoints and inspect responses after changes
- **pytest + FastAPI `TestClient` / Flask `test_client`** — route testing
  without a running server
- **Alembic** (if using SQLAlchemy) — generate/apply migrations; treat
  applying migrations as a "confirm before running" step even solo, since
  they're hard to undo cleanly
- **FastAPI's auto OpenAPI schema** (`/openapi.json`) — give the agent this
  as context instead of re-reading every route file to understand your API
  surface

---

## 5. ML/DL Engineering Tools

- **GPU visibility**: shell access to `nvidia-smi` so the agent checks
  VRAM/utilization before launching a training job (prevents OOM crashes
  it has no way to diagnose otherwise)
- **Experiment tracking**: MLflow (self-hosted, fully local) or Weights &
  Biases — lets the agent log runs and **read back past metrics**, so it
  can answer "did this change actually help" instead of guessing
- **Background job execution + log tailing** — training runs are long;
  agent should launch and detach, then check in later via log tail or
  tracking API
- **Checkpoint convention** — fixed directories (e.g. `models/`,
  `checkpoints/`) so the agent doesn't need to be told every time
- **Framework libs**: torch/tensorflow, scikit-learn, transformers, etc. —
  installed per-project venv, not global

---

## 6. Data Analysis Tools

- **pandas** and/or **polars** in the environment
- Persistent kernel (§2.5) is what makes this usable — avoids reloading
  large datasets every tool call
- Plot capture: matplotlib/seaborn figures returned as images from the
  kernel tool so you (and the agent) can actually see output
- Optional: **pandera** or **great_expectations** for data validation if
  pipelines need it — skip otherwise, adds overhead for personal use

---

## 7. Environment & Safety Setup

Even for solo local use, a few guardrails save you real pain:

- **Per-project venv or container** — isolates ML dependency hell
  (CUDA/torch version conflicts) from web project dependencies
- **`.gitignore` discipline** — prevent the agent from committing model
  checkpoints, large datasets, or `.env` files
- **Git checkpoint before agent sessions** — cheap insurance, instant rollback
- **Resource awareness** — agent should check GPU/RAM/disk before starting
  a new training job if one may already be running
- **Confirm-before-run list** — even in a personal tool, flag a few
  destructive-ish actions for manual confirmation rather than full auto-run:
  - `git push`
  - Applying DB migrations
  - Deleting files/directories
  - Anything that touches real (non-test) data
- **Secrets handling** — load API keys/DB credentials from `.env`, never
  let the agent print or log them; exclude `.env` from any context sent
  to the LLM

---

## 8. Agent Harness Requirements (the orchestration layer)

Beyond individual tools, the harness itself needs:

- **Tool-calling loop** — LLM proposes tool call → harness executes →
  result fed back → repeat until done
- **Context/memory management** — summarize old tool outputs, avoid
  blowing the context window with full file dumps or long logs
- **Self-verification loop** — after edits, automatically run
  lint → type-check → tests, feed failures back to the LLM to self-correct
- **Session logging** — record every tool call + result for your own
  debugging of the agent's behavior
- **Simple permission gate** — a config list of command patterns that
  require your manual "yes" (see §7) vs. ones that auto-run

---

## 9. Suggested Build Order

1. Shell tool + file read/write/edit + git — get a minimal loop working
2. Add ripgrep-based search
3. Add pytest/lint/type-check as post-edit verification steps
4. Add persistent Jupyter kernel tool for ML/data work
5. Add web search for doc lookups
6. Add background job launching + log tailing for training runs
7. Add MLflow integration for experiment read-back
8. Layer in the confirm-before-run gate and session logging last, once
   the core loop is trustworthy

---

## 10. Minimal Library List (Python side of the harness)

```
groq          # groq API client
jupyter_client      # persistent kernel for code execution tool
GitPython           # git tool (or just shell out to `git`)
python-dotenv        # secrets/env loading
uv (as CLI, not lib) # dependency management inside managed projects
```

Everything else (ripgrep, pytest, ruff, mypy, uvicorn, nvidia-smi, mlflow
CLI) is invoked via the shell tool rather than imported as a library —
keeps the harness itself simple and lets you swap tools per-project
without touching agent code.
   │
└───────────────────┬───────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │      Agent Harness       │  (you build this)
        │  - tool routing          │
        │  - context/memory mgmt   │
        │  - checkpointing         │
        └────────────┬────────────┘
                     │
   ┌─────────┬───────┼────────┬──────────┐
   │         │       │        │          │
 Shell     File I/O  Git   Python     Web Search
 Tool       Tool     Tool  Kernel      Tool
```

Core principle: keep the harness thin. Most "tools" should just be shell
commands the LLM invokes, not bespoke wrappers. The two exceptions worth
building custom tools for are **file editing** (patch-based) and a
**persistent Python execution environment** (for ML/data work).

---

## 2. Core Tools (build these first)

### 2.1 Shell/Command Execution Tool
- Executes arbitrary bash commands, returns stdout/stderr/exit code
- Runs inside a project-specific venv or container (see §6)
- Timeout + output truncation (long-running training jobs should be
  backgrounded, not block the tool call — see §2.5)

### 2.2 File Operations Tool
- `read_file(path, line_range=None)` — numbered line output for easy reference
- `write_file(path, content)` — create new files
- `edit_file(path, old_str, new_str)` — targeted patch, must match uniquely
- `list_dir(path, depth=2)` — directory tree, respecting `.gitignore`

Patch-based editing (not full-file rewrite) is important: saves tokens,
avoids clobbering unrelated code, and produces reviewable diffs.

### 2.3 Git Tool
- `git status/diff/log/branch/commit/checkout`
- Auto-commit checkpoint before agent makes changes, so every session is
  reversible with `git reset` or `git revert`
- Consider a dedicated branch per agent session (e.g. `agent/task-name`)
  so you can review before merging to `main`

### 2.4 Code Search Tool
- **ripgrep (`rg`)** for fast text/regex search across the repo
- Optional: tree-sitter or a language server (Pyright) for symbol-level
  search (go-to-definition, find-references) — much better than regex once
  your codebase grows past a few thousand lines

### 2.5 Persistent Python Execution Tool (most important custom tool)
For ML/data work, one-shot shell calls are wasteful (re-importing torch,
reloading a 2GB CSV every call). Build a tool backed by a **persistent
Jupyter kernel**:

- Use `jupyter_client` to start/manage a kernel per project
- `execute_code(code)` → returns stdout, stderr, rich outputs (dataframes,
  matplotlib images as base64, errors with traceback)
- Kernel state persists across calls within a session — variables, loaded
  models, dataframes all stay in memory
- For long-running training: launch as a **background process** via shell
  (`nohup`, `tmux`, or `subprocess.Popen`) instead of blocking the kernel,
  and give the agent a `tail_log(path, n_lines)` tool to check progress

### 2.6 Web Search / Doc Fetch Tool
- For current library docs, error message lookup, API references
- Avoids the agent guessing from stale training data on fast-moving
  libraries (transformers, langchain, fastapi, etc.)

---

## 3. Language & Package Tooling

| Purpose | Tool | Notes |
|---|---|---|
| Dependency mgmt | `uv` (preferred) or `poetry` | `uv` is much faster for repeated agent-driven installs |
| Env isolation | `venv` per project | Never let the agent install globally |
| Multiple Python versions | `pyenv` | If projects pin different versions |
| Linting | `ruff` | Fast, replaces flake8+isort+more |
| Formatting | `black` (or `ruff format`) | Run automatically after edits |
| Type checking | `mypy` or `pyright` | Pyright also gives you LSP-style navigation |
| Testing | `pytest` | Standard; agent should run this after every change |

---

## 4. Flask/FastAPI-Specific Tools

- **uvicorn** (FastAPI) / **flask run** — spin up the app for live testing
- **httpx / curl / httpie** — hit endpoints and inspect responses after changes
- **pytest + FastAPI `TestClient` / Flask `test_client`** — route testing
  without a running server
- **Alembic** (if using SQLAlchemy) — generate/apply migrations; treat
  applying migrations as a "confirm before running" step even solo, since
  they're hard to undo cleanly
- **FastAPI's auto OpenAPI schema** (`/openapi.json`) — give the agent this
  as context instead of re-reading every route file to understand your API
  surface

---

## 5. ML/DL Engineering Tools

- **GPU visibility**: shell access to `nvidia-smi` so the agent checks
  VRAM/utilization before launching a training job (prevents OOM crashes
  it has no way to diagnose otherwise)
- **Experiment tracking**: MLflow (self-hosted, fully local) or Weights &
  Biases — lets the agent log runs and **read back past metrics**, so it
  can answer "did this change actually help" instead of guessing
- **Background job execution + log tailing** — training runs are long;
  agent should launch and detach, then check in later via log tail or
  tracking API
- **Checkpoint convention** — fixed directories (e.g. `models/`,
  `checkpoints/`) so the agent doesn't need to be told every time
- **Framework libs**: torch/tensorflow, scikit-learn, transformers, etc. —
  installed per-project venv, not global

---

## 6. Data Analysis Tools

- **pandas** and/or **polars** in the environment
- Persistent kernel (§2.5) is what makes this usable — avoids reloading
  large datasets every tool call
- Plot capture: matplotlib/seaborn figures returned as images from the
  kernel tool so you (and the agent) can actually see output
- Optional: **pandera** or **great_expectations** for data validation if
  pipelines need it — skip otherwise, adds overhead for personal use

---

## 7. Environment & Safety Setup

Even for solo local use, a few guardrails save you real pain:

- **Per-project venv or container** — isolates ML dependency hell
  (CUDA/torch version conflicts) from web project dependencies
- **`.gitignore` discipline** — prevent the agent from committing model
  checkpoints, large datasets, or `.env` files
- **Git checkpoint before agent sessions** — cheap insurance, instant rollback
- **Resource awareness** — agent should check GPU/RAM/disk before starting
  a new training job if one may already be running
- **Confirm-before-run list** — even in a personal tool, flag a few
  destructive-ish actions for manual confirmation rather than full auto-run:
  - `git push`
  - Applying DB migrations
  - Deleting files/directories
  - Anything that touches real (non-test) data
- **Secrets handling** — load API keys/DB credentials from `.env`, never
  let the agent print or log them; exclude `.env` from any context sent
  to the LLM

---

## 8. Agent Harness Requirements (the orchestration layer)

Beyond individual tools, the harness itself needs:

- **Tool-calling loop** — LLM proposes tool call → harness executes →
  result fed back → repeat until done
- **Context/memory management** — summarize old tool outputs, avoid
  blowing the context window with full file dumps or long logs
- **Self-verification loop** — after edits, automatically run
  lint → type-check → tests, feed failures back to the LLM to self-correct
- **Session logging** — record every tool call + result for your own
  debugging of the agent's behavior
- **Simple permission gate** — a config list of command patterns that
  require your manual "yes" (see §7) vs. ones that auto-run

---

## 9. Suggested Build Order

1. Shell tool + file read/write/edit + git — get a minimal loop working
2. Add ripgrep-based search
3. Add pytest/lint/type-check as post-edit verification steps
4. Add persistent Jupyter kernel tool for ML/data work
5. Add web search for doc lookups
6. Add background job launching + log tailing for training runs
7. Add MLflow integration for experiment read-back
8. Layer in the confirm-before-run gate and session logging last, once
   the core loop is trustworthy

---

## 10. Minimal Library List (Python side of the harness)

```
groq          # groq API client
jupyter_client      # persistent kernel for code execution tool
GitPython           # git tool (or just shell out to `git`)
python-dotenv        # secrets/env loading
uv (as CLI, not lib) # dependency management inside managed projects
```

Everything else (ripgrep, pytest, ruff, mypy, uvicorn, nvidia-smi, mlflow
CLI) is invoked via the shell tool rather than imported as a library —
keeps the harness itself simple and lets you swap tools per-project
without touching agent code.
