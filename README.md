# Local Coding Agent

An extensible, local-first coding assistant built with **LangGraph**, **LangChain**, and **Pydantic**. This agent is designed for Python backend, ML engineering, and general software development tasks. It features a multi-tiered LLM fallback system, interactive human-in-the-loop confirmation gates for destructive operations, persistent Jupyter kernel state, and detailed JSON-lines event logging.

---

## 📌 Project Summary

The **Local Coding Agent** acts as an automated coding assistant that inspects, edits, tests, and verifies code within your project environment. Instead of rewriting whole files, it prefers targeted context editing (`edit_file`) and chunked file creation (`append_file`). 

### Core Capabilities
* **Multi-Tier Model Fallback:** Tries **Groq** $\rightarrow$ **OpenRouter** $\rightarrow$ local **Ollama** (`qwen2.5-coder:14b`) seamlessly.
* **Human-in-the-Loop Safety:** Destructive actions (e.g., shell commands, background process launches, file deletions, git pushes) require explicit confirmation before execution.
* **Persistent Python Execution:** Interactive code execution inside isolated Jupyter kernels for data science and ML workloads.
* **Repository & File Management:** Scans directory trees, respects `.gitignore`, runs `ripgrep` searches, and performs patch edits.
* **Automated Git Workflows:** Stages changes, handles commits, checks diffs, and creates feature branches.
* **Observability & Resilience:** Automatic context window trimming, rate-limit retry logic, tool-hallucination auto-healing, and event logging.

---

## 🛠️ Workflow & Architecture

The execution lifecycle is modeled as a state machine using **LangGraph**.

```
                   +------------------------+
                   |       START            |
                   +-----------+------------+
                               |
                               v
                       +---------------+
                       |  agent_node   | <----------------+
                       +-------+-------+                  |
                               |                          |
                   +-----------v-----------+              |
                   |  route_after_agent    |              |
                   +---+---------------+---+              |
                       |               |                  |
    (Tool Call Needs   |               | (Safe Tool Call) |
      Confirmation)    |               |                  |
                       v               v                  |
                 +----------+    +------------+           |
                 | confirm  | --->|   tools    +----------+
                 |  _node   |    |   _node    |
                 +----------+    +------------+
                       |
               (Terminal / Max Iterations / Error)
                       |
                       v
                   +---+---+
                   |  END  |
                   +-------+
```

### Detailed Workflow Steps

1. **Initialization (`run_agent`):**
   * Loads configuration from environment variables (`.env`).
   * Binds tools from `ALL_TOOLS` to the LLM model client.
   * Generates or attaches to a unique `thread_id`.

2. **Agent Node (`agent_node`):**
   * Summarizes and prepares message context (collapses old tool outputs and slims large call parameters to fit model token bounds).
   * Calls the configured provider chain (Groq $\rightarrow$ OpenRouter $\rightarrow$ Ollama).
   * **Auto-Healing Rules:** Handles rate limiting with exponential backoff, recovers from leaked text JSON tool calls, and re-prompts on empty responses or tool hallucinations.

3. **Routing (`route_after_agent`):**
   * If the LLM generates a final text response or hits the maximum iteration count (`max_iterations`), the workflow terminates.
   * If the LLM requests a tool call, the router determines whether human confirmation is required.

4. **Human Confirmation Node (`confirm_node`):**
   * Pauses the graph using `langgraph.types.interrupt`.
   * Triggers the CLI input handler (or a custom UI callback).
   * If approved, the workflow proceeds to execution. If declined, a message is injected back into the LLM context to explain the block and request an alternative plan.

5. **Tool Execution Node (`tools_node`):**
   * Executes the requested tool function.
   * Truncates outputs exceeding character limits (1,200 characters by default) to keep token usage efficient.
   * Logs execution status and results to `agent_events.log`.
   * Loops back to `agent_node` with the new tool results.

---

## 🔒 Safety & Confirmation Rules

To prevent accidental data loss or unauthorized remote calls, specific tools trigger an interactive confirmation gate:

| Tool | Trigger Condition | Danger Level |
|---|---|---|
| `run_shell_command` | Always | High |
| `git_push` | Requires `confirm=True` & Human Approval | High |
| `delete_path` | Requires `confirm=True` & Human Approval | High |
| `launch_background_process` | Always | Medium |
| `write_file` | When `overwrite=True` | Medium |

---

## 📦 Requirements & Dependencies

### Prerequisites
* **Python:** 3.10 or higher
* **Optional System Dependencies:**
  * `ripgrep` (`rg`): Required for `ripgrep_search` tool.
  * `Ollama`: Required if local model fallback mode is enabled.
  * `nvidia-smi`: Required for `check_gpu_status`.

### Core Python Packages

```text
pydantic>=2.0
langgraph
langchain-core
langchain-openai
langchain-ollama
jupyter_client
httpx
python-dotenv
pathspec               # Optional: for gitignore parsing in directory listing
duckduckgo-search     # Optional: for web_search tool
```

---

## ⚙️ Environment & Configuration

Create a `.env` file in the root directory:

```ini
# Primary Provider (Groq)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_TEMPERATURE=0.1
GROQ_MAX_TOKENS=4096
GROQ_REQUEST_TIMEOUT=60.0
GROQ_MAX_RETRIES=3

# Fallback Tier 1 (OpenRouter) - Optional
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-oss-120b:free

# Fallback Tier 2 (Local Ollama) - Optional
AGENT_ENABLE_OLLAMA_FALLBACK=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:14b
OLLAMA_NUM_CTX=4096

# Agent Controls
AGENT_PROVIDER_MODE=api        # Options: 'api' or 'local'
AGENT_MAX_ITERATIONS=20
AGENT_LOG_FILE=agent_events.log
```

---

## 🛠️ Tool Reference

The agent comes equipped with 20 pre-built tools across 6 operational categories:

### 1. File Operations (`file_tool.py`)
* `read_file(path, start_line, end_line)`: Reads files with line numbers; caps output at 300 lines by default.
* `write_file(path, content, overwrite)`: Creates new files; requires approval if overwriting.
* `append_file(path, content, create_if_missing)`: Appends data; ideal for building large files in chunks.
* `edit_file(path, old_str, new_str)`: Replaces exact, unique code snippets.
* `list_dir(path, depth)`: Displays folder trees while honoring `.gitignore`.

### 2. Version Control (`git_tool.py`)
* `git_status`: Displays branch information and uncommitted changes.
* `git_diff`: Shows staged/unstaged changes.
* `git_log`: Lists recent commit history.
* `git_branch`: Lists or creates branches.
* `git_checkout`: Switches branches or restores files.
* `git_commit`: Stages and commits changes.
* `git_push`: Pushes branch to remote (`confirm=True` required).

### 3. Code Execution & Notebooks (`jupyter_tool.py`)
* `execute_code(code, project_id)`: Runs Python inside a persistent Jupyter kernel. Preserves variables across calls.
* `restart_kernel(project_id)`: Resets state and frees VRAM/RAM.

### 4. Search & Discovery (`search_tool.py`)
* `ripgrep_search(pattern, path, file_type)`: Fast code search via `rg`.
* `web_search(query)`: Fetches up-to-date online documentation via DuckDuckGo.

### 5. System & Process Management (`shell_tool.py`, `system_tool.py`)
* `run_shell_command(command, cwd, timeout)`: Runs CLI commands (tests, linters).
* `check_gpu_status`: Queries `nvidia-smi` for GPU VRAM and usage.
* `launch_background_process(command, job_name)`: Runs long-running tasks in the background.
* `tail_log(job_name, n_lines)`: Inspects background task output logs.
* `delete_path(path, confirm)`: Deletes files or folders (`confirm=True` required).

### 6. API Helpers (`api_tool.py`)
* `http_request(url, method, json_body, headers)`: Tests local or remote HTTP endpoints.
* `fetch_openapi_schema(base_url)`: Retrieves OpenAPI JSON specs from web endpoints (e.g., FastAPI/Flask).

---

## 🚀 Quickstart Usage

### Basic Python Usage

```python
from agent import run_agent

# Run a single task
result = run_agent("Inspect the files in this repository and list all TODO comments")

print(f"Status: {result.status}")
print(f"Iterations: {result.iterations}")
print(f"Output:\n{result.output}")
```

### Resume / Multi-Turn Session

```python
from agent import run_agent

# First prompt
res1 = run_agent("Create a new git branch called feature/test-suite")

# Continue conversation using the same thread_id
res2 = run_agent(
    "Now write a test file test_app.py using pytest",
    thread_id=res1.thread_id
)
```

### Custom Confirmation Handler (e.g., Web/API Integration)

```python
from agent import run_agent
from agent.schemas import ConfirmationDecision, ConfirmationRequest

def custom_ui_handler(req: ConfirmationRequest) -> ConfirmationDecision:
    # Wire this up to a Web UI, Discord bot, or Slack prompt
    print(f"Approval Request for {req.tool_name} with args {req.tool_args}")
    # Auto-approve or trigger webhook response
    return ConfirmationDecision(approved=True)

result = run_agent(
    "Run pytest on the repository",
    confirm_handler=custom_ui_handler
)
```

---

## 📝 Observability & Logging

Every run automatically writes event lines to the log file configured by `AGENT_LOG_FILE` (`agent_events.log` by default). Each entry is formatted as a JSON line:

```json
{"timestamp": 1774567890.12, "event": "run_start", "thread_id": "8a32b2f...", "prompt": "Inspect repo...", "provider_mode": "api"}
{"timestamp": 1774567891.45, "event": "llm_call", "thread_id": "8a32b2f...", "provider": "groq:openai/gpt-oss-120b"}
{"timestamp": 1774567892.10, "event": "tool_call", "thread_id": "8a32b2f...", "tool_name": "list_dir", "args": {"path": "."}, "success": true, "result": "config.py\ngraph.py..."}
{"timestamp": 1774567895.80, "event": "run_end", "thread_id": "8a32b2f...", "status": "completed", "iterations": 2, "tool_call_count": 1, "error": null}
```