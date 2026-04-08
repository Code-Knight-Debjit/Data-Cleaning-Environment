"""
inference.py
------------
Data Cleaning Pipeline — submission inference script.

Supports:
  • Ollama local llama3 (DEFAULT — no API key needed)
  • Groq free cloud API
  • Any OpenAI-compatible endpoint

Environment variables:
    API_BASE_URL     LLM endpoint.   Default: http://localhost:11434/v1  (Ollama)
    MODEL_NAME       Model name.     Default: llama3
    HF_TOKEN         API key.        Default: "ollama" (ignored by Ollama)
    LOCAL_IMAGE_NAME Docker image    (leave unset to use ENV_BASE_URL)
    ENV_BASE_URL     Env server URL. Default: http://localhost:8000

To switch to Groq instead of Ollama:
    $env:API_BASE_URL = "https://api.groq.com/openai/v1"
    $env:MODEL_NAME   = "llama-3.3-70b-versatile"
    $env:HF_TOKEN     = "gsk_xxxxxxxxxxxx"

STDOUT FORMAT (evaluator parses exactly — do not modify):
    [START] task=<n> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from client import DataCleaningEnv
    from models import CleanAction, MAX_STEPS, DONE_THRESHOLD
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from client import DataCleaningEnv
    from models import CleanAction, MAX_STEPS, DONE_THRESHOLD


# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL     = os.getenv("API_BASE_URL",     "http://localhost:11434/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",       "llama3")
HF_TOKEN         = os.getenv("HF_TOKEN",         "ollama")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL",     "http://localhost:8000")

BENCHMARK   = "data_cleaning_env"
TASK_IDS    = ["easy", "medium", "hard"]
STEP_LIMITS = {"easy": 40, "medium": 100, "hard": 150}


# ── System prompt (deterministic agent) ──────────────────────────────────────

SYSTEM_PROMPT = """You are a deterministic data cleaning agent.
Your task is to clean a dataset step-by-step using valid actions.
You are operating inside an environment with strict rules.
--------------------------------------------------
## INPUT PROVIDED EACH STEP
You will receive:
1. Column schema (LIST OF VALID COLUMN NAMES - CASE SENSITIVE)
2. Column status:
   - missing values count
   - whether standardized (true/false)
3. Remaining issues (global state)
4. Previous actions taken
--------------------------------------------------
## OBJECTIVE
Fully clean the dataset with MINIMUM steps.
A dataset is CLEAN only if:
- No missing values remain
- All columns are standardized
- No invalid formats exist
--------------------------------------------------
## STRICT RULES (MUST FOLLOW)

### 1. NEVER TERMINATE EARLY
You MUST NOT output DONE unless:
- ALL columns have missing = 0
- ALL columns have standardized = true
- remaining_issues is empty
If ANY issue remains -> DO NOT output DONE.

### 2. USE ONLY VALID COLUMNS
- You MUST use EXACT column names from the schema list
- Column names are CASE SENSITIVE
- NEVER invent new column names

### 3. PRIORITIZE COLUMN-LEVEL ACTIONS
Preferred actions (in order):
  1. FILL_MISSING    - fixes entire column missing values
  2. STANDARDIZE_COL - fixes formatting for entire column
  3. SET_VALUE        - only for a single isolated bad cell
  4. DROP_ROW         - only for truly corrupt/outlier rows
NEVER fix a full column using repeated SET_VALUE.

### 4. DO NOT REPEAT ACTIONS
- Do NOT apply the same action to the same column twice
- Do NOT standardize an already standardized column
- Do NOT fill missing if missing = 0

### 5. CHOOSE THE CORRECT FILL STRATEGY
- Numeric columns (float/int): use "median" or "mean"
- Categorical/string columns: use "mode"
- NEVER use "mean" or "median" on a categorical column

### 6. ALWAYS THINK GLOBALLY
Before choosing an action:
- Review ALL columns in column_status
- Pick the single action that fixes the largest remaining issue
--------------------------------------------------
## DECISION PROCESS (MANDATORY)
At each step:
1. Read column_status carefully
2. Find columns where missing > 0 OR standardized = false
3. If none exist AND remaining_issues is empty -> output DONE
4. Otherwise, pick the ONE most impactful action
--------------------------------------------------
## OUTPUT FORMAT - STRICT JSON ONLY
Return ONLY a single JSON object. No explanation. No markdown. No backticks.

Fill missing values:
{"action": "FILL_MISSING", "column": "<exact_col_name>", "strategy": "<mean|median|mode>"}

Standardize a column:
{"action": "STANDARDIZE_COL", "column": "<exact_col_name>"}

Fix one cell:
{"action": "SET_VALUE", "column": "<exact_col_name>", "row": <int>, "value": "<str>"}

Drop a bad row:
{"action": "DROP_ROW", "row": <int>}

Signal completion:
{"action": "DONE"}

--------------------------------------------------
## FAILURE CONDITIONS (YOU WILL BE PENALIZED FOR):
- Outputting DONE when issues remain
- Using a column name not in the schema
- Repeating the same action on the same column
- Using SET_VALUE to fix an entire column
- Using mean/median on a categorical column
- Using mode on a numeric column
--------------------------------------------------
## FINAL GOAL
Be efficient, precise, and minimal.
Every step must move the dataset closer to a fully clean state."""


# ── Official log helpers ──────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action[:80].replace(chr(10), ' ')} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ── Column type hints (used to suggest fill strategies) ──────────────────────

_COL_TYPES: Dict[str, Dict[str, str]] = {
    "easy": {
        "order_id":   "numeric",
        "customer":   "categorical",
        "product":    "categorical",
        "category":   "categorical",
        "price":      "numeric",
        "quantity":   "numeric",
        "order_date": "datetime",
        "region":     "categorical",
    },
    "medium": {
        "tx_id":       "numeric",
        "customer_id": "numeric",
        "amount":      "numeric",
        "tx_date":     "datetime",
        "category":    "categorical",
        "country":     "categorical",
        "status":      "categorical",
    },
    "hard": {
        "record_id":     "numeric",   "id":            "numeric",   "RecordID":    "numeric",
        "customer_id":   "numeric",   "cust_id":       "numeric",   "CustomerID":  "numeric",
        "full_name":     "categorical","name":          "categorical","CustomerName":"categorical",
        "email":         "categorical","email_address": "categorical","Email":       "categorical",
        "amount":        "numeric",   "sale_amount":   "numeric",   "Amount":      "numeric",
        "currency":      "categorical","ccy":           "categorical","Currency":    "categorical",
        "purchase_date": "datetime",  "date":          "datetime",  "PurchaseDate":"datetime",
        "product_name":  "categorical","item":          "categorical","ProductName": "categorical",
        "region":        "categorical","territory":     "categorical","area":        "categorical",
        "contact_email": "categorical","value":         "numeric",   "product":     "categorical",
    },
}


def _strategy_hint(task_id: str, col: str) -> str:
    col_type = _COL_TYPES.get(task_id, {}).get(col, "unknown")
    if col_type == "numeric":
        return "median"
    if col_type in ("categorical", "datetime"):
        return "mode"
    return "median"


# ── Prompt builder ────────────────────────────────────────────────────────────

def _column_status_block(obs, task_id: str) -> str:
    col_status: Dict[str, Any] = getattr(obs, "column_status", {}) or {}

    if col_status:
        lines = []
        for col, status in col_status.items():
            missing      = status.get("missing", 0)
            standardized = status.get("standardized", True)
            hint         = _strategy_hint(task_id, col)
            flag         = "OK" if (missing == 0 and standardized) else "NEEDS_FIX"
            lines.append(
                f"  {col:<22} missing={missing:<4} "
                f"standardized={str(standardized).lower():<5}  "
                f"fill_strategy={hint:<7}  [{flag}]"
            )
        return "\n".join(lines)

    # Fallback: derive columns from CSV header
    rows   = obs.dirty_csv.strip().split("\n")
    header = rows[0] if rows else ""
    cols   = [c.strip() for c in header.split(",")]
    return "\n".join(
        f"  {col:<22} (status unknown)  fill_strategy={_strategy_hint(task_id, col)}"
        for col in cols
    )


def build_user_prompt(obs, history: List[str]) -> str:
    rows      = obs.dirty_csv.strip().split("\n")
    header    = rows[0] if rows else ""
    data_rows = rows[1:]
    preview   = "\n".join([header] + data_rows[:10])
    truncated = len(data_rows) > 10

    col_status: Dict[str, Any] = getattr(obs, "column_status", {}) or {}
    broken = [
        c for c, s in col_status.items()
        if s.get("missing", 0) > 0 or not s.get("standardized", True)
    ]

    history_block = (
        "\n".join(f"  {h}" for h in history[-6:])
        if history else "  (none yet)"
    )

    return (
        f"--------------------------------------------------\n"
        f"## STEP {obs.step_number}/{obs.max_steps}\n"
        f"Score:            {obs.current_score:.4f}  "
        f"(need >= {DONE_THRESHOLD[obs.task_id]:.2f} to pass)\n"
        f"Issues remaining: {obs.issues_remaining}\n"
        f"Broken columns:   {len(broken)} -> {broken[:10] if broken else 'NONE — consider DONE'}\n"
        f"\n## SCHEMA HINT\n{obs.schema_hint}\n"
        f"\n## VALID COLUMN NAMES (CASE SENSITIVE — copy exactly)\n{header}\n"
        f"\n## COLUMN STATUS (read carefully before acting)\n"
        f"{_column_status_block(obs, obs.task_id)}\n"
        f"\n## CSV PREVIEW"
        f"{' (first 10 of ' + str(len(data_rows)) + ' rows)' if truncated else ''}\n"
        f"{preview}\n"
        f"\n## PREVIOUS ACTIONS (last 6)\n{history_block}\n"
        f"\n--------------------------------------------------\n"
        f"## DECISION CHECKLIST\n"
        f"1. Any column with missing > 0?  -> FILL_MISSING (use strategy from column status)\n"
        f"2. Any column with standardized=false?  -> STANDARDIZE_COL\n"
        f"3. Isolated bad cell visible in CSV?  -> SET_VALUE\n"
        f"4. Clearly corrupt/outlier row?  -> DROP_ROW\n"
        f"5. ALL missing=0, ALL standardized=true, issues=0?  -> DONE\n"
        f"\nOutput ONE JSON action (no markdown, no explanation):"
    )


# ── Action parser ─────────────────────────────────────────────────────────────
# Bridges {action, column, strategy, row, value} -> CleanAction

_COMMAND_MAP = {
    "FILL_MISSING":    "FILL_MISSING",
    "STANDARDIZE_COL": "STANDARDIZE_COL",
    "STANDARDIZE":     "STANDARDIZE_COL",
    "SET_VALUE":       "SET_VALUE",
    "DROP_ROW":        "DROP_ROW",
    "DROP":            "DROP_ROW",
    "DONE":            "DONE",
}


def parse_action(raw: str) -> CleanAction:
    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        text  = "\n".join(inner).strip()

    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return CleanAction(command="DONE")

    try:
        data: Dict[str, Any] = json.loads(m.group())
    except json.JSONDecodeError:
        return CleanAction(command="DONE")

    raw_cmd = str(data.get("action", "DONE")).upper().strip().replace(" ", "_")
    command = _COMMAND_MAP.get(raw_cmd)
    if not command:
        return CleanAction(command="DONE")
    if command == "DONE":
        return CleanAction(command="DONE")

    column        = data.get("column")
    fill_strategy = data.get("strategy") or data.get("fill_strategy")
    row_raw       = data.get("row") if data.get("row") is not None else data.get("row_index")
    value         = data.get("value")

    try:
        return CleanAction(
            command       = command,
            column        = column,
            fill_strategy = fill_strategy,
            row_index     = int(row_raw) if row_raw is not None else None,
            value         = str(value) if value is not None else None,
        )
    except Exception:
        return CleanAction(command="DONE")


# ── LLM call (async — keeps WebSocket keepalive alive) ───────────────────────

async def call_llm_async(client: OpenAI, messages: list) -> str:
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = messages,
            max_tokens  = 120,
            temperature = 0.0,
        ),
    )
    return (response.choices[0].message.content or "").strip()


# ── Episode loop ───────────────────────────────────────────────────────────────

async def run_episode(env, client: OpenAI, task_id: str) -> dict:
    max_steps        = STEP_LIMITS[task_id]
    threshold        = DONE_THRESHOLD[task_id]
    rewards: List[float] = []
    steps_taken      = 0
    score            = 0.0
    success          = False
    history: List[str] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs    = result.observation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            steps_taken = step
            messages.append({"role": "user", "content": build_user_prompt(obs, history)})

            try:
                raw    = await call_llm_async(client, messages)
                action = parse_action(raw)
                messages.append({"role": "assistant", "content": raw})
            except Exception as exc:
                log_step(step, "DONE", 0.00, True, str(exc)[:120])
                rewards.append(0.0)
                break

            # Keep system + last 3 exchanges to avoid context overflow
            if len(messages) > 7:
                messages = [messages[0]] + messages[-6:]

            result = await env.step(action)
            obs    = result.observation
            reward = result.reward or 0.0
            rewards.append(reward)
            score  = obs.current_score

            log_step(
                step   = step,
                action = action.command,
                reward = reward,
                done   = obs.done,
                error  = obs.last_action_error,
            )

            err_note = f" [ERR: {obs.last_action_error[:40]}]" if obs.last_action_error else ""
            history.append(
                f"step {step}: {action.command}"
                + (f"({action.column}"
                   + (f", {action.fill_strategy})" if action.fill_strategy else ")")
                   if action.column else "")
                + f" -> score={score:.4f}{err_note}"
            )

            if obs.done or score >= threshold:
                break

        success = score >= threshold

    except Exception as e:
        print(f"[EPISODE ERROR] task={task_id} error={str(e)[:120]}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score":   score,
        "reward":  sum(rewards),
        "steps":   steps_taken,
        "success": success,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    is_ollama = "11434" in API_BASE_URL or "ollama" in API_BASE_URL.lower()

    if not is_ollama and (not HF_TOKEN or HF_TOKEN == "ollama"):
        print(
            "ERROR: HF_TOKEN not set for remote API.\n"
            "For Groq:  $env:HF_TOKEN='gsk_xxxxxxxxxxxx'\n"
            "For Ollama (local): no token needed — defaults already set.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"MODEL_NAME   : {MODEL_NAME}",   flush=True)
    print(f"BACKEND      : {'Ollama (local)' if is_ollama else 'Remote API'}", flush=True)
    print(f"ENV SERVER   : {LOCAL_IMAGE_NAME or ENV_BASE_URL}", flush=True)
    print("", flush=True)

    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    results = []
    for task_id in TASK_IDS:
        # Fresh connection per task — prevents WebSocket keepalive timeout carryover
        if LOCAL_IMAGE_NAME:
            env = await DataCleaningEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env = DataCleaningEnv(base_url=ENV_BASE_URL)
            await env.connect()

        try:
            summary = await run_episode(env, llm, task_id)
            results.append(summary)
        finally:
            try:
                await env.close()
            except Exception:
                pass
        print("", flush=True)

    print("=" * 56, flush=True)
    print(f"{'Task':<10} {'Score':>7} {'Reward':>9} {'Steps':>6} {'Pass':>5}")
    print("-" * 56, flush=True)
    for r in results:
        print(
            f"{r['task_id']:<10} {r['score']:>7.4f} {r['reward']:>9.4f} "
            f"{r['steps']:>6}  {'YES' if r['success'] else 'NO':>4}",
            flush=True,
        )
    print("=" * 56, flush=True)


if __name__ == "__main__":
    asyncio.run(main())