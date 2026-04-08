"""
inference.py
------------
Official submission inference script for the Data Cleaning Pipeline environment.

Environment variables (all free — no paid API):
    API_BASE_URL       LLM endpoint.  Default: HuggingFace free router.
    MODEL_NAME         Model to use.  Default: Qwen/Qwen2.5-72B-Instruct (free).
    HF_TOKEN           Your free HuggingFace token (hf_...).
    LOCAL_IMAGE_NAME   Docker image name if using from_docker_image() — leave
                       unset to use ENV_BASE_URL instead.
    ENV_BASE_URL       Server URL. Default: http://localhost:8000

STDOUT FORMAT (evaluator parses these lines — do not modify):
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


# ── Configuration — all defaults are FREE ─────────────────────────────────────

API_BASE_URL     = os.getenv("API_BASE_URL",     "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",       "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN",         "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL",     "http://localhost:8000")

BENCHMARK  = "data_cleaning_env"
TASK_IDS   = ["easy", "medium", "hard"]
STEP_LIMITS = {"easy": 25, "medium": 50, "hard": 80}


# ── System prompt (expert data cleaning agent) ────────────────────────────────

SYSTEM_PROMPT = """You are an expert data cleaning agent operating in a structured environment.
Your goal is to transform the dataset into a fully clean state using the minimum number of steps.
You are given:
1. Column schema (with data types)
2. Column status:
   - missing values count
   - whether the column is standardized
3. Remaining issues (global view)
4. Previous actions

---
## STRICT RULES

### 1. DO NOT terminate early
You MUST NOT output DONE unless ALL of the following are true:
- No missing values remain in any column
- All columns are standardized
- No formatting issues remain
- No invalid values exist
If ANY issue remains → continue acting.

---
### 2. Prioritize column-level fixes
Prefer:
- FILL_MISSING (for missing values)
- STANDARDIZE_COL (for formatting / normalization)
Avoid:
- SET_VALUE (only use for isolated anomalies)
NEVER fix an entire column using repeated SET_VALUE.

---
### 3. Use correct strategies based on column type
- Numeric columns → mean or median
- Categorical columns → mode
- Datetime columns → STANDARDIZE_COL (not SET_VALUE unless single anomaly)

---
### 4. Do not repeat work
- Do NOT standardize a column more than once unless state changed
- Do NOT fill missing if missing = 0

---
### 5. Always reason about global completion
Before choosing DONE, check:
- column_status
- remaining_issues
If any column has:
- missing > 0
- standardized = false
→ DO NOT choose DONE

---
## DECISION PROCESS (MANDATORY)
At each step:
1. Identify remaining issues
2. Select the MOST impactful action
3. Prefer actions that resolve entire columns
4. Avoid redundant or low-value actions

---
## OUTPUT FORMAT
Return ONLY a valid JSON action — no explanation, no markdown fences:

For column-level fixes:
{"action": "FILL_MISSING", "column": "<col>", "strategy": "<mean|median|mode>"}
{"action": "STANDARDIZE_COL", "column": "<col>"}

For isolated cell fixes:
{"action": "SET_VALUE", "column": "<col>", "row": <int>, "value": "<str>"}

For outlier rows:
{"action": "DROP_ROW", "row": <int>}

When everything is clean:
{"action": "DONE"}

---
## OBJECTIVE
Minimize: number of steps, redundant operations, row-level edits.
Maximize: completeness, correctness, efficiency.

You will be penalized for: premature DONE, repeated actions, unnecessary SET_VALUE usage.

Think step-by-step internally, but ONLY output the final JSON action."""


# ── Official log format ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action[:80].replace(chr(10),' ')} "
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


# ── Prompt builder ────────────────────────────────────────────────────────────

def _format_column_status(column_status: Dict[str, Any]) -> str:
    """Render column_status as a compact, agent-readable block."""
    if not column_status:
        return "  (not available)"
    lines = []
    for col, status in column_status.items():
        missing     = status.get("missing", 0)
        standardized = status.get("standardized", True)
        issues      = status.get("issues", [])
        flag = "OK" if missing == 0 and standardized else "NEEDS_FIX"
        issue_str = ", ".join(issues) if issues else ""
        lines.append(
            f"  {col:<22} missing={missing} standardized={str(standardized).lower()}"
            + (f" issues=[{issue_str}]" if issue_str else "")
            + f" → {flag}"
        )
    return "\n".join(lines)


def build_user_prompt(obs, history: List[str]) -> str:
    rows      = obs.dirty_csv.strip().split("\n")
    header    = rows[0] if rows else ""
    preview   = "\n".join(rows[:25])
    truncated = len(rows) > 25

    col_status_block = _format_column_status(
        getattr(obs, "column_status", {})
    )

    history_block = (
        "\n".join(f"  {h}" for h in history[-6:]) if history else "  (none yet)"
    )

    # Count truly broken columns
    col_status = getattr(obs, "column_status", {})
    broken = [
        c for c, s in col_status.items()
        if s.get("missing", 0) > 0 or not s.get("standardized", True)
    ]

    return f"""## Current State
Task:              {obs.task_id}
Step:              {obs.step_number}/{obs.max_steps}
Score:             {obs.current_score:.4f}  (need {DONE_THRESHOLD[obs.task_id]:.2f} for success)
Issues remaining:  {obs.issues_remaining}
Broken columns:    {len(broken)} → {broken[:8]}

## Schema hint
{obs.schema_hint}

## Column status
{col_status_block}

## CSV columns
{header}

## CSV preview{' (first 25 rows)' if truncated else ''}
{preview}

## Previous actions (last 6)
{history_block}

## Your task
Select the single most impactful action to bring broken columns to clean state.
Check column_status — if all columns show missing=0 and standardized=true → output DONE.
Otherwise → pick the highest-impact fix.
Return ONLY valid JSON, no markdown."""


# ── Action parsing ─────────────────────────────────────────────────────────────
# The system prompt uses {action, column, strategy, row, value}.
# CleanAction uses  {command, column, fill_strategy, row_index, value}.
# This function bridges the two.

def parse_action(raw: str) -> CleanAction:
    text = raw.strip()

    # Strip markdown fences if model wraps output
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        text  = "\n".join(inner).strip()

    # Extract first {...} block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return CleanAction(command="DONE")

    try:
        data: Dict[str, Any] = json.loads(match.group())
    except json.JSONDecodeError:
        return CleanAction(command="DONE")

    # ── Field mapping: prompt format → CleanAction format ─────────────────
    action_name: str = str(data.get("action", "DONE")).upper().replace(" ", "_")

    if action_name == "DONE":
        return CleanAction(command="DONE")

    # Normalise command name (prompt may say FILL_MISSING, STANDARDIZE_COL, etc.)
    command_map = {
        "FILL_MISSING":    "FILL_MISSING",
        "STANDARDIZE_COL": "STANDARDIZE_COL",
        "STANDARDIZE":     "STANDARDIZE_COL",
        "SET_VALUE":       "SET_VALUE",
        "DROP_ROW":        "DROP_ROW",
        "DROP":            "DROP_ROW",
    }
    command = command_map.get(action_name)
    if command is None:
        return CleanAction(command="DONE")

    column        = data.get("column")
    # "strategy" in prompt → "fill_strategy" in CleanAction
    fill_strategy = data.get("strategy") or data.get("fill_strategy")
    # "row" in prompt → "row_index" in CleanAction
    row_index     = data.get("row") if data.get("row") is not None else data.get("row_index")
    value         = data.get("value")

    try:
        return CleanAction(
            command=command,
            column=column,
            fill_strategy=fill_strategy,
            row_index=int(row_index) if row_index is not None else None,
            value=str(value) if value is not None else None,
        )
    except Exception:
        return CleanAction(command="DONE")


def call_llm(client: OpenAI, messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=150,
        temperature=0.0,   # deterministic — the prompt is already very directive
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
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            steps_taken = step
            user_msg    = build_user_prompt(obs, history)
            messages.append({"role": "user", "content": user_msg})

            try:
                raw    = call_llm(client, messages)
                action = parse_action(raw)
                messages.append({"role": "assistant", "content": raw})
            except Exception as exc:
                log_step(step, "DONE", 0.00, True, str(exc)[:120])
                rewards.append(0.0)
                break

            # Keep system + last 10 turns (5 user + 5 assistant) inside context
            if len(messages) > 21:
                messages = [messages[0]] + messages[-20:]

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

            # Track history for agent context
            err_note = f" [BLOCKED: {obs.last_action_error[:50]}]" \
                       if obs.last_action_error else ""
            history.append(
                f"step {step}: {action.command}"
                + (f" col={action.column}" if action.column else "")
                + (f" strategy={action.fill_strategy}" if action.fill_strategy else "")
                + f" → score={score:.4f}{err_note}"
            )

            if obs.done or score >= threshold:
                break

        success = score >= threshold

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score,
            "reward": sum(rewards), "steps": steps_taken, "success": success}


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    if not HF_TOKEN:
        print(
            "ERROR: HF_TOKEN is not set.\n"
            "1. Go to https://huggingface.co/settings/tokens\n"
            "2. Click 'New token' → 'Read' access → copy the hf_... token\n"
            "3. In PowerShell: $env:HF_TOKEN='hf_xxxxxxxxxxxx'\n"
            "4. Run: python inference.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"MODEL_NAME   : {MODEL_NAME}",   flush=True)
    print(f"TARGET       : {LOCAL_IMAGE_NAME or ENV_BASE_URL}", flush=True)
    print("", flush=True)

    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    if LOCAL_IMAGE_NAME:
        env = await DataCleaningEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = DataCleaningEnv(base_url=ENV_BASE_URL)
        await env.connect()

    results = []
    try:
        for task_id in TASK_IDS:
            summary = await run_episode(env, llm, task_id)
            results.append(summary)
            print("", flush=True)
    finally:
        await env.close()

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