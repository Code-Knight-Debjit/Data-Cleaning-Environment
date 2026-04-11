"""
inference.py
------------
Official submission inference script for the Data Cleaning Pipeline environment.

Environment variables:
    API_BASE_URL   LLM endpoint.  Default: HuggingFace free router.
    MODEL_NAME     Model to use.  Default: Qwen/Qwen2.5-72B-Instruct (free).
    HF_TOKEN       Your HuggingFace token (hf_...).
    ENV_BASE_URL   The running environment URL.
                   Set this to your HuggingFace Space URL, e.g.:
                   https://CodeKnightDebjit-data-cleaning-env.hf.space

NOTE: Do NOT use LOCAL_IMAGE_NAME / from_docker_image() in submitted scripts.
The evaluator machine does not have your local Docker image — it connects to
your live HF Space via ENV_BASE_URL.

STDOUT FORMAT (evaluator parses these exactly):
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


# ── Configuration ──────────────────────────────────────────────────────────────
# ENV_BASE_URL must point to your live HuggingFace Space.
# The evaluator sets this automatically when it runs your script.

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://CodeKnightDebjit-data-cleaning-env.hf.space")

BENCHMARK   = "data_cleaning_env"
TASK_IDS    = ["easy", "medium", "hard"]
STEP_LIMITS = {"easy": 25, "medium": 50, "hard": 80}


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a deterministic data cleaning agent.
Your task is to clean a dataset step-by-step using valid actions.
You are operating inside an environment with strict rules.
--------------------------------------------------
## INPUT PROVIDED EACH STEP
You will receive:
1. Column schema (LIST OF VALID COLUMN NAMES — CASE SENSITIVE)
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
If ANY issue remains → DO NOT output DONE.
--------------------------------------------------
### 2. USE ONLY VALID COLUMNS
- You MUST use EXACT column names from schema
- Column names are CASE SENSITIVE
- NEVER invent new column names
--------------------------------------------------
### 3. PRIORITIZE COLUMN-LEVEL ACTIONS
Preferred actions:
- FILL_MISSING (fixes entire column)
- STANDARDIZE_COL (fixes formatting)
Avoid:
- SET_VALUE (only for single isolated errors)
NEVER fix a full column using repeated SET_VALUE.
--------------------------------------------------
### 4. DO NOT REPEAT ACTIONS
- Do NOT apply the same action repeatedly on the same column
- Do NOT standardize an already standardized column
- Do NOT fill missing if missing = 0
--------------------------------------------------
### 5. AVOID DESTRUCTIVE ACTIONS
- DROP_ROW should be used ONLY when absolutely necessary
--------------------------------------------------
## OUTPUT FORMAT (STRICT JSON ONLY)
Return ONLY one of these — no explanation, no markdown:
{"action": "FILL_MISSING", "column": "<col>", "strategy": "<mean|median|mode>"}
{"action": "STANDARDIZE_COL", "column": "<col>"}
{"action": "SET_VALUE", "column": "<col>", "row": <int>, "value": "<str>"}
{"action": "DROP_ROW", "row": <int>}
{"action": "DONE"}
--------------------------------------------------
## FAILURE CONDITIONS (AVOID THESE)
- DONE prematurely → penalty -1.0
- Invalid column names → action fails
- Repeated same action → wasted step
--------------------------------------------------
Every step must move the dataset closer to a fully clean state."""


# ── Official log format ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action[:80].replace(chr(10), ' ')} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _col_status_block(column_status: Dict[str, Any]) -> str:
    if not column_status:
        return "  (not available)"
    lines = []
    for col, s in column_status.items():
        missing      = s.get("missing", 0)
        standardized = s.get("standardized", True)
        issues       = s.get("issues", [])
        flag         = "OK" if (missing == 0 and standardized) else "NEEDS_FIX"
        issue_str    = ", ".join(issues) if issues else ""
        lines.append(
            f"  {col:<26} missing={missing:<3} standardized={str(standardized).lower():<5}"
            + (f"  issues=[{issue_str}]" if issue_str else "")
            + f"  → {flag}"
        )
    return "\n".join(lines)


def build_user_prompt(obs, history: List[str]) -> str:
    col_status: Dict[str, Any] = getattr(obs, "column_status", {})
    valid_columns = list(col_status.keys())
    broken = [c for c, s in col_status.items()
              if s.get("missing", 0) > 0 or not s.get("standardized", True)]

    rows    = obs.dirty_csv.strip().split("\n")
    preview = "\n".join(rows[:21])

    all_clean = len(broken) == 0
    done_hint = (
        "ALL columns clean → you MAY output DONE"
        if all_clean else
        f"{len(broken)} column(s) still broken → DO NOT output DONE"
    )

    history_block = "\n".join(f"  {h}" for h in history[-6:]) if history else "  none"

    return f"""--------------------------------------------------
## COLUMN SCHEMA (EXACT CASE-SENSITIVE NAMES — USE THESE EXACTLY)
{chr(10).join(f'  - {c}' for c in valid_columns)}

--------------------------------------------------
## COLUMN STATUS
{_col_status_block(col_status)}

--------------------------------------------------
## GLOBAL STATE
Task:             {obs.task_id}
Step:             {obs.step_number} / {obs.max_steps}
Score:            {obs.current_score:.4f}  (need >= {DONE_THRESHOLD[obs.task_id]:.2f})
Remaining issues: {obs.issues_remaining}
Broken columns:   {broken}
DONE status:      {done_hint}

--------------------------------------------------
## SCHEMA HINT
{obs.schema_hint}

--------------------------------------------------
## CSV PREVIEW (first 20 rows)
{preview}

--------------------------------------------------
## PREVIOUS ACTIONS
{history_block}

--------------------------------------------------
Return ONLY valid JSON — no explanation, no markdown."""


# ── Action parsing ─────────────────────────────────────────────────────────────

COMMAND_MAP = {
    "FILL_MISSING":    "FILL_MISSING",
    "STANDARDIZE_COL": "STANDARDIZE_COL",
    "STANDARDIZE":     "STANDARDIZE_COL",
    "SET_VALUE":       "SET_VALUE",
    "DROP_ROW":        "DROP_ROW",
    "DROP":            "DROP_ROW",
}

VALID_STRATEGIES = {"mean", "median", "mode", "drop"}


def parse_action(raw: str, valid_columns: List[str]) -> CleanAction:
    text = raw.strip()
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

    action_raw = str(data.get("action", "DONE")).strip().upper().replace(" ", "_")

    if action_raw == "DONE":
        return CleanAction(command="DONE")

    command = COMMAND_MAP.get(action_raw)
    if command is None:
        return CleanAction(command="DONE")

    # Validate column name (case-sensitive, with case-insensitive fallback)
    column = data.get("column")
    if column is not None and valid_columns:
        if column not in valid_columns:
            col_lower = {c.lower(): c for c in valid_columns}
            column = col_lower.get(str(column).lower())  # None if no match

    # strategy → fill_strategy
    fill_strategy = data.get("strategy") or data.get("fill_strategy")
    if fill_strategy and str(fill_strategy).lower() not in VALID_STRATEGIES:
        fill_strategy = "median"

    # row → row_index
    row_raw = data.get("row") if data.get("row") is not None else data.get("row_index")
    row_index = None
    if row_raw is not None:
        try:
            row_index = int(row_raw)
        except (TypeError, ValueError):
            pass

    value = data.get("value")

    try:
        return CleanAction(
            command       = command,
            column        = column,
            fill_strategy = fill_strategy,
            row_index     = row_index,
            value         = str(value) if value is not None else None,
        )
    except Exception:
        return CleanAction(command="DONE")


def call_llm(client: OpenAI, messages: list) -> str:
    response = client.chat.completions.create(
        model       = MODEL_NAME,
        messages    = messages,
        max_tokens  = 100,
        temperature = 0.0,
    )
    return (response.choices[0].message.content or "").strip()


# ── Episode runner ─────────────────────────────────────────────────────────────

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

        valid_columns: List[str] = list(getattr(obs, "column_status", {}).keys())
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            steps_taken = step
            messages.append({"role": "user", "content": build_user_prompt(obs, history)})

            try:
                raw    = call_llm(client, messages)
                action = parse_action(raw, valid_columns)
                messages.append({"role": "assistant", "content": raw})
            except Exception as exc:
                log_step(step, "DONE", 0.00, True, str(exc)[:120])
                rewards.append(0.0)
                break

            # Keep system + last 10 turns inside free-tier context limit
            if len(messages) > 21:
                messages = [messages[0]] + messages[-20:]

            result = await env.step(action)
            obs    = result.observation

            if getattr(obs, "column_status", {}):
                valid_columns = list(obs.column_status.keys())

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

            parts = [f"step {step}: {action.command}"]
            if action.column:       parts.append(f"col={action.column}")
            if action.fill_strategy: parts.append(f"strategy={action.fill_strategy}")
            parts.append(f"score={score:.4f}")
            if obs.last_action_error:
                parts.append(f"[BLOCKED: {obs.last_action_error[:60]}]")
            history.append("  ".join(parts))

            if obs.done or score >= threshold:
                break

        success = score >= threshold

    except Exception as episode_err:
        # Catch-all so [END] is always emitted even if the episode crashes
        print(f"[DEBUG] Episode error: {episode_err}", flush=True)
        log_end(success=False, steps=steps_taken, score=score, rewards=rewards)
        return {"task_id": task_id, "score": score, "reward": sum(rewards),
                "steps": steps_taken, "success": False}

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "reward": sum(rewards),
            "steps": steps_taken, "success": success}


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    if not HF_TOKEN:
        print(
            "ERROR: HF_TOKEN is not set.\n"
            "1. Go to https://huggingface.co/settings/tokens\n"
            "2. Create a Read token and copy it\n"
            "3. Set it:  $env:HF_TOKEN='hf_xxxxxxxxxxxx'  (PowerShell)\n"
            "            export HF_TOKEN='hf_xxxxxxxxxxxx'  (bash)\n"
            "4. Run:     python inference.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"MODEL_NAME   : {MODEL_NAME}",   flush=True)
    print(f"ENV_BASE_URL : {ENV_BASE_URL}", flush=True)
    print("", flush=True)

    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Always connect via URL — no Docker on the evaluator machine
    env = DataCleaningEnv(base_url=ENV_BASE_URL)
    await env.connect()

    results = []
    try:
        for task_id in TASK_IDS:
            summary = await run_episode(env, llm, task_id)
            results.append(summary)
            print("", flush=True)
    finally:
        try:
            await env.close()
        except Exception:
            pass

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