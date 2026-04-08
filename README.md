<div align="center">

# 🧹 Data Cleaning Environment

### A Reinforcement Learning Benchmark for Autonomous Data Cleaning Agents

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-FF6B35?style=for-the-badge)](https://github.com/meta-pytorch/OpenEnv)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployable-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **An OpenEnv-compatible reinforcement learning environment where an LLM agent receives a dirty CSV dataset and must autonomously fix type errors, outliers, missing values, and schema inconsistencies to match a hidden ground truth — step by step.**

<br/>

```
┌──────────────────────────────────────────────────────────────────┐
│   Dirty CSV  →  Agent Observes  →  Issues CleanAction  →  Reward │
│                                                                  │
│   "N/A"  →  FILL_MISSING(median)  →  Score ↑  →  +0.12 reward  │
│   "2099" →  SET_VALUE(row=3,"2024-01-15")  →  Score ↑  →  +0.08 │
│   "  bob" → STANDARDIZE_COL("name")  →  Score ↑  →  +0.05       │
└──────────────────────────────────────────────────────────────────┘
```

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tasks](#-tasks)
- [Action Space](#-action-space)
- [Observation Space](#-observation-space)
- [Reward Function](#-reward-function)
- [Quick Start](#-quick-start)
- [Running Inference](#-running-inference)
- [Environment API](#-environment-api)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Development & Testing](#-development--testing)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 Overview

The **Data Cleaning Environment** is a structured RL benchmark where an LLM-powered agent must clean tabular datasets. The environment wraps a FastAPI WebSocket server following the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) protocol, making it compatible with any OpenEnv-based training or evaluation framework.

### Why This Matters

Real-world data pipelines spend 60–80% of their time on data cleaning. This environment trains agents to:

- **Detect** type errors, outliers, missing values, and schema inconsistencies
- **Reason** about which fix is most impactful at each step
- **Self-correct** from informative error feedback
- **Terminate** efficiently without over-cleaning

### Key Properties

| Property | Value |
|---|---|
| Protocol | OpenEnv (WebSocket + HTTP) |
| Action Space | Discrete (5 command types) |
| Observation | Full CSV state + grader feedback |
| Episode Structure | Reset → N × Step → Done |
| Concurrency | ✅ Multiple simultaneous sessions |
| State Management | Server-side, fully isolated per session |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent (LLM / RL Policy)                     │
│                  Qwen2.5-72B / Mistral / Custom Model               │
└────────────────────────┬───────────────────────────────┬────────────┘
                         │ CleanAction (JSON)             │ CleanObservation
                         ▼                               │
┌────────────────────────────────────────────────────────┴────────────┐
│                      DataCleaningEnv (client.py)                     │
│               OpenEnv EnvClient[CleanAction, CleanObservation, dict] │
│                   WebSocket persistent connection                     │
└────────────────────────┬────────────────────────────────────────────┘
                         │  WebSocket /ws
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Server (server/app.py)                  │
│                  HTTP + WebSocket endpoints, sessions                │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│               DataCleaningEnvironment (server/data_cleaning_env.py)  │
│                                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────┐ │
│  │ dataset_    │  │  Action      │  │  Grader   │  │  Reward    │ │
│  │ factory.py  │  │  Dispatcher  │  │  Engine   │  │  Computer  │ │
│  │             │  │  SET_VALUE   │  │  grade()  │  │            │ │
│  │ easy/medium │  │  DROP_ROW    │  │  score    │  │  progress  │ │
│  │ /hard CSVs  │  │  STANDARD.   │  │  delta    │  │  efficiency│ │
│  │             │  │  FILL_MISS.  │  │           │  │  penalties │ │
│  └─────────────┘  └──────────────┘  └───────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
data_cleaning_env/
│
├── 📄 client.py                  # DataCleaningEnv — OpenEnv client
├── 📄 models.py                  # CleanAction, CleanObservation, CleanState (Pydantic)
├── 📄 inference.py               # Official evaluation entry point
├── 📄 dataset_factory.py         # Generates easy/medium/hard dirty↔clean CSV pairs
├── 📄 graders.py                 # Scoring engine — grade(agent_df vs clean_df)
├── 📄 openenv.yaml               # OpenEnv manifest (HuggingFace Spaces config)
├── 📄 pyproject.toml             # Project metadata and dependencies
│
└── server/
    ├── 📄 app.py                 # FastAPI application (HTTP + WebSocket)
    ├── 📄 data_cleaning_env.py   # Core environment logic (reset/step/state)
    ├── 📄 __init__.py
    └── 📄 Dockerfile             # Container image definition
```

---

## 🎯 Tasks

The environment ships three progressively harder tasks, each with fixed-seed deterministic datasets:

### 🟢 Easy — Sales Orders

| Property | Value |
|---|---|
| Dataset | ~100-row sales orders CSV |
| Dirty Issues | Cell-level type errors, a few missing values |
| Step Budget | **40 steps** |
| Success Threshold | **Score ≥ 0.95** |
| Primary Skills | `SET_VALUE`, `FILL_MISSING` |

**What the agent needs to fix:** Individual cells with wrong types (e.g., `"N/A"` in a price column, `"abc"` in a numeric field). Straightforward injected errors with clear ground truth.

---

### 🟡 Medium — Financial Transactions

| Property | Value |
|---|---|
| Dataset | ~200-row transaction log |
| Dirty Issues | Outlier rows, mixed date formats, missing amounts |
| Step Budget | **80 steps** |
| Success Threshold | **Score ≥ 0.85** |
| Primary Skills | `DROP_ROW`, `STANDARDIZE_COL`, `FILL_MISSING` |

**What the agent needs to fix:** Statistical outliers disguised as data, inconsistent date formats, missing numeric values. Crucially, some extreme values are **valid** — dropping them costs a false-positive penalty.

---

### 🔴 Hard — Multi-Schema Dataset

| Property | Value |
|---|---|
| Dataset | ~400-row multi-domain CSV |
| Dirty Issues | Cross-column inconsistencies, future-year dates, bulk missing data |
| Step Budget | **150 steps** |
| Success Threshold | **Score ≥ 0.80** |
| Primary Skills | All commands |

**What the agent needs to fix:** Everything from easy + medium, plus cascading schema issues across columns. Requires strategic planning about fix order.

---

## 🕹️ Action Space

Every step the agent sends exactly one `CleanAction`:

```python
from models import CleanAction

# Fix a specific cell
CleanAction(command="SET_VALUE", row_index=3, column="price", value="29.99")

# Remove an entire row (use carefully — false positives are penalised)
CleanAction(command="DROP_ROW", row_index=17)

# Normalise a column's format (dates → YYYY-MM-DD, numbers → float, strings → stripped)
CleanAction(command="STANDARDIZE_COL", column="order_date")

# Fill all NaN values in a column using a strategy
CleanAction(command="FILL_MISSING", column="quantity", fill_strategy="median")

# Signal episode completion (only accepted when score ≥ task threshold)
CleanAction(command="DONE")
```

### Command Reference

| Command | `row_index` | `column` | `value` | `fill_strategy` |
|---|---|---|---|---|
| `SET_VALUE` | ✅ required | ✅ required | ✅ required | — |
| `DROP_ROW` | ✅ required | — | — | — |
| `STANDARDIZE_COL` | — | ✅ required | — | — |
| `FILL_MISSING` | — | ✅ required | — | ✅ required |
| `DONE` | — | — | — | — |

### `FILL_MISSING` Strategies

| Strategy | Behaviour |
|---|---|
| `"mean"` | Replace NaN with column mean (numeric columns only) |
| `"median"` | Replace NaN with column median (numeric columns only) |
| `"mode"` | Replace NaN with most frequent value (any column) |
| `"drop"` | Remove rows where this column is NaN |

> ⚠️ **Important:** `DROP_ROW` removes by **positional row index** (the `row_index` column in the CSV), not by a row ID field. Row indices shift after each drop.

---

## 👁️ Observation Space

After every `reset()` and `step()`, the agent receives a `CleanObservation`:

```python
@dataclass
class CleanObservation:
    # ── Task context (constant per episode) ──────────────────────
    task_id: str               # "easy" | "medium" | "hard"
    schema_hint: str           # Plain-English description of clean schema
    initial_dirty_cells: int   # Total dirty cells at episode start

    # ── Per-step state ───────────────────────────────────────────
    dirty_csv: str             # Full current CSV as string (all edits applied)
    current_score: float       # 0.0 → 1.0  (grader score vs ground truth)
    issues_remaining: int      # Approximate dirty cells still to fix
    step_number: int           # Steps taken so far
    max_steps: int             # Budget for this task

    # ── Last-action feedback ─────────────────────────────────────
    last_action_success: bool  # Whether previous action applied cleanly
    last_action_error: str     # Error message if success=False (else None)

    # ── Inherited ────────────────────────────────────────────────
    done: bool                 # True = episode ended
    reward: float | None       # Per-step reward (None after reset)
```

### Score Computation

The grader compares the agent's working DataFrame to the hidden ground-truth DataFrame:

```
score = (initial_dirty_cells - remaining_dirty_cells) / initial_dirty_cells
```

A score of `1.0` means perfect agreement with ground truth.

---

## 💰 Reward Function

The reward is dense and shaped to guide efficient, precise cleaning:

```
reward = progress_term
       + efficiency_bonus
       + false_positive_penalty
       + early_done_penalty
       + step_cost
```

| Component | Value | When |
|---|---|---|
| **Progress** | `current_score − previous_score` | Every step |
| **Efficiency bonus** | `+0.10 × (1 − steps_used/max_steps)` | Only when task is solved this step |
| **False-positive penalty** | `−0.15` | `DROP_ROW` removes a valid-extreme row (medium task) |
| **Early DONE penalty** | `−0.20` | `DONE` called with score < 0.60 |
| **Step cost** | `−0.005` | Every step (discourages padding) |
| **Premature DONE block** | `−1.00` | `DONE` below task threshold — episode *continues* |

**Reward range:** `[−0.5, +1.0]` (clipped)

### Termination Logic

The episode terminates when **any** of these is true:

1. ✅ `current_score >= task_threshold` (auto-terminated, efficiency bonus awarded)
2. ✅ Agent sends `DONE` and `current_score >= task_threshold` (accepted)
3. ⏱️ `step_count >= max_steps` (budget exhausted)

`DONE` is **refused** if the score is below threshold — the episode continues with a `−1.0` reward signal.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker Desktop (for containerised server)
- A free [HuggingFace token](https://huggingface.co/settings/tokens) (for the inference LLM)

### 1. Clone & Install

```bash
git clone https://github.com/Code-Knight-Debjit/Data-Cleaning-Environment.git
cd Data-Cleaning-Environment

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 2. Build the Docker Image

```bash
docker build -t openenv-data_cleaning:latest -f server/Dockerfile .
```

### 3. Set Your HuggingFace Token

```powershell
# Windows PowerShell
$env:HF_TOKEN = "hf_your_token_here"

# macOS / Linux
export HF_TOKEN="hf_your_token_here"
```

### 4. Run Inference

```bash
python inference.py
```

That's it! The script auto-starts the Docker container, runs the LLM agent through all three tasks (easy → medium → hard), and prints structured evaluation logs.

---

## 🤖 Running Inference

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(required)* | Your HuggingFace token for LLM API access |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use for inference |
| `LOCAL_IMAGE_NAME` | `openenv-data_cleaning:latest` | Docker image to launch |
| `ENV_BASE_URL` | `http://localhost:8000` | Direct server URL (if not using Docker) |

### Switching Models

```powershell
# Use Mistral (smaller, faster)
$env:MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Use Llama
$env:MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
```

### Connecting to a Running Server (skip Docker)

```powershell
$env:LOCAL_IMAGE_NAME = ""   # must be empty string
$env:ENV_BASE_URL = "http://localhost:8000"
python inference.py
```

### Expected Output

```
API_BASE_URL     : https://router.huggingface.co/v1
MODEL_NAME       : Qwen/Qwen2.5-72B-Instruct
LOCAL_IMAGE_NAME : openenv-data_cleaning:latest
ENV_BASE_URL     : http://localhost:8000

[START] task=easy env=data_cleaning_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1  action=FILL_MISSING  reward=0.12 done=false  error=null
[STEP]  step=2  action=SET_VALUE     reward=0.08 done=false  error=null
[STEP]  step=3  action=STANDARDIZE_COL reward=0.05 done=false error=null
...
[END]   success=true steps=18 score=0.97 rewards=0.12,0.08,...

[START] task=medium env=data_cleaning_env ...
...

════════════════════════════════════════════════════════
Task        Score    Reward  Steps  Pass
────────────────────────────────────────────────────────
easy       0.9712    1.3400     18   YES
medium     0.8823    2.1100     47   YES
hard       0.7640    1.8500     98    NO
════════════════════════════════════════════════════════
```

---

## 🔌 Environment API

### Using the Python Client Directly

```python
import asyncio
from client import DataCleaningEnv
from models import CleanAction

async def run():
    # Option A: Auto-start Docker container
    env = await DataCleaningEnv.from_docker_image("openenv-data_cleaning:latest")

    # Option B: Connect to an already-running server
    # env = DataCleaningEnv(base_url="http://localhost:8000")
    # await env.connect()

    try:
        # Reset for a specific task
        result = await env.reset(task_id="easy")
        obs = result.observation

        print(f"Score: {obs.current_score:.4f}")
        print(f"Issues: {obs.issues_remaining}")
        print(f"Schema: {obs.schema_hint}")

        # Take a step
        action = CleanAction(
            command="FILL_MISSING",
            column="price",
            fill_strategy="median"
        )
        result = await env.step(action)
        obs = result.observation

        print(f"Reward: {result.reward:.4f}")
        print(f"New score: {obs.current_score:.4f}")
        print(f"Action OK: {obs.last_action_success}")

        # Signal completion
        result = await env.step(CleanAction(command="DONE"))

    finally:
        await env.close()

asyncio.run(run())
```

### Using the Sync Wrapper

```python
from client import DataCleaningEnv
from models import CleanAction

env = DataCleaningEnv(base_url="http://localhost:8000").sync()

with env:
    result = env.reset(task_id="easy")
    result = env.step(CleanAction(command="STANDARDIZE_COL", column="order_date"))
    print(f"Score: {result.observation.current_score:.4f}")
```

### HTTP Endpoints

When the server is running, the following HTTP endpoints are available:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health check |
| `/docs` | GET | Swagger / OpenAPI documentation |
| `/web` | GET | Interactive web UI |
| `/ws` | WebSocket | Persistent session endpoint |

---

## ⚙️ Configuration

### Step Budgets

```python
MAX_STEPS = {
    "easy":   40,
    "medium": 80,
    "hard":   150,
}
```

### Success Thresholds

```python
DONE_THRESHOLD = {
    "easy":   0.95,
    "medium": 0.85,
    "hard":   0.80,
}
```

### Reward Constants

| Constant | Value | Purpose |
|---|---|---|
| `STEP_COST` | `-0.005` | Per-step penalty to discourage padding |
| `EARLY_DONE_PENALTY` | `-0.20` | Penalty for `DONE` below score 0.60 |
| `EARLY_DONE_THRESHOLD` | `0.60` | Score floor for DONE without penalty |
| `FALSE_POSITIVE_PENALTY` | `-0.15` | Penalty for wrongly dropping a valid row |
| `EFFICIENCY_BONUS_WEIGHT` | `0.10` | Multiplier for early-completion bonus |

---

## ☁️ Deployment

### Deploy to HuggingFace Spaces

```bash
# Install the OpenEnv CLI
pip install openenv

# Authenticate with HuggingFace
huggingface-cli login

# Deploy (from the repo root where openenv.yaml lives)
openenv push

# Or deploy privately to a specific repo
openenv push --repo-id your-username/data-cleaning-env --private
```

After deployment, your environment will be live at:
```
https://huggingface.co/spaces/your-username/data-cleaning-env
```

With endpoints:
- **Web UI:** `/web`
- **API Docs:** `/docs`
- **Health:** `/health`
- **WebSocket:** `/ws`

### Connect to a HuggingFace Space

```python
env = await DataCleaningEnv.from_env("your-username/data-cleaning-env")
# or run locally with UV (no Docker needed)
env = await DataCleaningEnv.from_env("your-username/data-cleaning-env", use_docker=False)
```

### Run the Server Locally (Without Docker)

```bash
uvicorn server.app:app --reload --port 8000
```

---

## 🧪 Development & Testing

### Test the Environment Logic (No Server Needed)

```bash
# Runs a smoke test across all three tasks
python server/data_cleaning_env.py
```

Expected output:
```
────────────────────────────────────────────────────────────────
TASK: EASY
────────────────────────────────────────────────────────────────
reset()  → score=0.0000  issues=29  done=False
  CSV:  101 rows, 5 cols
  Hint: Sales orders dataset. price must be float...
step (bad col) → success=False  error='Column 'DOES_NOT_EXIST' not found...'
step (fix row=3 col='price') → success=True  score=0.0345  reward=0.0295
step (DONE, blocked)  → done=False  reward=-1.0  score=0.0345
...
All smoke tests passed.
```

### Test Pydantic Models

```bash
python models.py
```

### Test the Client Parser

```bash
python test_parse.py
```

### Run the Full Server Locally

```bash
uvicorn server.app:app --reload
# Open http://localhost:8000/docs for interactive API explorer
```

---

## 🔧 Troubleshooting

### `TypeError: Too few arguments for EnvClient`

**Cause:** Your `client.py` subclasses `EnvClient` with only 2 type parameters, but OpenEnv requires 3 (`ActT`, `ObsT`, `StateT`).

**Fix:**
```python
# ❌ Wrong
class DataCleaningEnv(EnvClient[CleanAction, CleanObservation]):

# ✅ Correct
class DataCleaningEnv(EnvClient[CleanAction, CleanObservation, dict]):
```

Also ensure `_parse_state` is implemented:
```python
def _parse_state(self, payload: dict) -> dict:
    return payload
```

---

### `ValidationError: Input should be 'SET_VALUE', 'DROP_ROW', ...`

**Cause:** Passing an invalid command string to `CleanAction`.

**Fix:** Only these 5 commands are valid:
```python
"SET_VALUE" | "DROP_ROW" | "STANDARDIZE_COL" | "FILL_MISSING" | "DONE"
```
There is no `"drop_column"` — columns cannot be dropped, only rows.

---

### `UnboundLocalError: cannot access local variable 'env'`

**Cause 1:** Docker image doesn't exist yet.
```bash
docker build -t openenv-data_cleaning:latest -f server/Dockerfile .
```

**Cause 2:** Stray test lines in `inference.py` referencing `env` before it's assigned.

**Fix:** Remove any manually added lines like `action = CleanAction(...)` or `result = await env.step(action)` from inside `main()`. The `main()` function should only call `run_episode()` — all action logic belongs inside that function.

---

### `DONE rejected: score X < required Y`

**This is expected behaviour, not a bug.** The environment refuses premature termination. The agent should continue cleaning until the score meets the task threshold.

---

### HuggingFace Router returns 401

Ensure your token is set:
```powershell
$env:HF_TOKEN = "hf_your_token_here"
```
Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## 📐 Data Flow Diagram

```
                    ┌──────────────────────────────────┐
                    │   inference.py / custom agent    │
                    │                                  │
                    │  1. await env.reset(task_id=…)   │
                    │  2. obs = result.observation      │
                    │  3. build_prompt(obs) → LLM       │
                    │  4. parse_action(llm_output)      │
                    │  5. await env.step(action)        │
                    │  6. GOTO 2 until done             │
                    └──────────────┬───────────────────┘
                                   │
                    CleanAction (JSON over WebSocket)
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │        DataCleaningEnvironment    │
                    │                                  │
                    │  _apply_action()                 │
                    │    → mutates _dirty_df in-place  │
                    │                                  │
                    │  grade(agent_df vs clean_df)     │
                    │    → score ∈ [0.0, 1.0]          │
                    │                                  │
                    │  _compute_reward()               │
                    │    → progress + bonuses          │
                    │                                  │
                    │  _build_observation()            │
                    │    → CleanObservation            │
                    └──────────────────────────────────┘
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Run the smoke tests: `python server/data_cleaning_env.py`
4. Commit your changes: `git commit -m "feat: add my improvement"`
5. Push and open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ using [OpenEnv](https://github.com/meta-pytorch/OpenEnv) · [FastAPI](https://fastapi.tiangolo.com/) · [Pydantic](https://docs.pydantic.dev/) · [HuggingFace](https://huggingface.co/)

</div>