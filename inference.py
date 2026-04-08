"""
Inference Script — Data Cleaning Environment
=============================================
MANDATORY environment variables:
    API_BASE_URL       The API endpoint for the LLM.
    MODEL_NAME         The model identifier to use for inference.
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   Docker image name (when using from_docker_image()).

Defaults are set only for API_BASE_URL and MODEL_NAME (not HF_TOKEN).

STDOUT FORMAT
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import DataCleaningEnv
from models import CleanAction

# ── Environment variables ────────────────────────────────────────────────────
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "openenv-data_cleaning:latest")
API_KEY          = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK        = "data_cleaning_env"

# ── Per-task config (mirrors server constants) ────────────────────────────────
TASK_CONFIG = {
    "easy":   {"max_steps": 40,  "threshold": 0.95},
    "medium": {"max_steps": 80,  "threshold": 0.85},
    "hard":   {"max_steps": 150, "threshold": 0.80},
}

TEMPERATURE = 0.2   # low temp → more deterministic action parsing
MAX_TOKENS  = 256

# ── Logging helpers (strict stdout format) ───────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Prompt builders ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert data cleaning agent. You receive a dirty CSV dataset and must
    fix it step by step to match a hidden clean ground truth.

    Available commands (respond with EXACTLY one JSON object, no extra text):

    {"command": "SET_VALUE",       "row_index": <int>, "column": "<col>", "value": "<val>"}
    {"command": "DROP_ROW",        "row_index": <int>}
    {"command": "STANDARDIZE_COL", "column": "<col>"}
    {"command": "FILL_MISSING",    "column": "<col>", "fill_strategy": "mean|median|mode|drop"}
    {"command": "DONE"}

    Rules:
    - Output ONLY the JSON object — no explanation, no markdown, no backticks.
    - Use DONE only when you are confident the score meets the task threshold.
    - SET_VALUE fixes a single bad cell.
    - STANDARDIZE_COL normalises an entire column's format.
    - FILL_MISSING fills NaN values in a column.
    - DROP_ROW removes a row; use carefully — false positives are penalised.
    - Row indices are 0-based positional indices (they shift after each DROP_ROW).
""").strip()


def build_user_prompt(obs, history: List[str]) -> str:
    history_block = "\n".join(history[-15:]) if history else "None yet."
    return textwrap.dedent(f"""
        Task: {obs.task_id}
        Schema hint: {obs.schema_hint}
        Step: {obs.step_number} / {obs.max_steps}
        Current score: {obs.current_score:.4f}
        Issues remaining: {obs.issues_remaining}
        Initial dirty cells: {obs.initial_dirty_cells}
        Last action success: {obs.last_action_success}
        Last action error: {obs.last_action_error or 'none'}

        === ACTION HISTORY (most recent 15) ===
        {history_block}

        IMPORTANT RULES:
        - Do NOT repeat any action that already appears in the history with score_delta=0.0000.
        - Do NOT repeat STANDARDIZE_COL or FILL_MISSING on the same column twice.
        - If score is not improving after 2 steps, switch strategy entirely.
        - Use SET_VALUE to fix specific bad cells (wrong types, "N/A" strings, outliers, future dates).
        - Inspect the CSV carefully before choosing your action.

        Current CSV (first 80 rows shown if large):
        {_truncate_csv(obs.dirty_csv, max_rows=80)}

        Output your next CleanAction as a single JSON object.
    """).strip()


def _truncate_csv(csv_text: str, max_rows: int = 80) -> str:
    lines = csv_text.splitlines()
    if len(lines) <= max_rows + 1:   # +1 for header
        return csv_text
    header = lines[0]
    body   = lines[1: max_rows + 1]
    omitted = len(lines) - 1 - max_rows
    return "\n".join([header] + body + [f"... ({omitted} more rows omitted)"])

# ── Action parsing ────────────────────────────────────────────────────────────

VALID_COMMANDS = {"SET_VALUE", "DROP_ROW", "STANDARDIZE_COL", "FILL_MISSING", "DONE"}
VALID_STRATEGIES = {"mean", "median", "mode", "drop"}


def parse_action(llm_output: str) -> CleanAction:
    """
    Parse the LLM's JSON output into a CleanAction.
    Falls back to STANDARDIZE_COL on the first column if parsing fails.
    """
    text = llm_output.strip()

    # Strip accidental markdown fences
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # Extract first JSON object
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {text!r}")

    data = json.loads(match.group())
    command = data.get("command", "").upper()

    if command not in VALID_COMMANDS:
        raise ValueError(f"Unknown command: {command!r}")

    if command == "SET_VALUE":
        return CleanAction(
            command="SET_VALUE",
            row_index=int(data["row_index"]),
            column=str(data["column"]),
            value=str(data["value"]),
        )
    elif command == "DROP_ROW":
        return CleanAction(command="DROP_ROW", row_index=int(data["row_index"]))
    elif command == "STANDARDIZE_COL":
        return CleanAction(command="STANDARDIZE_COL", column=str(data["column"]))
    elif command == "FILL_MISSING":
        strategy = str(data.get("fill_strategy", "median")).lower()
        if strategy not in VALID_STRATEGIES:
            strategy = "median"
        return CleanAction(
            command="FILL_MISSING",
            column=str(data["column"]),
            fill_strategy=strategy,
        )
    else:  # DONE
        return CleanAction(command="DONE")


def _action_to_str(action: CleanAction) -> str:
    """Compact single-line string for [STEP] log."""
    parts = [action.command]
    if action.row_index is not None:
        parts.append(f"row={action.row_index}")
    if action.column:
        parts.append(f"col={action.column}")
    if action.value is not None:
        val_repr = str(action.value)[:30]
        parts.append(f"val={val_repr!r}")
    if action.fill_strategy:
        parts.append(f"strategy={action.fill_strategy}")
    return "(" + ",".join(parts) + ")"

# ── LLM call ──────────────────────────────────────────────────────────────────

def get_model_action(client: OpenAI, obs, history: List[str]) -> CleanAction:
    user_prompt = build_user_prompt(obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_action(text)
    except Exception as exc:
        print(f"[DEBUG] Model/parse error: {exc}", flush=True)
        return CleanAction(command="FILL_MISSING", column="quantity", fill_strategy="median")

# ── Episode runner ────────────────────────────────────────────────────────────

async def run_episode(env: DataCleaningEnv, client: OpenAI, task_id: str) -> dict:
    """
    Run a single episode for task_id. Returns a summary dict.
    """
    cfg        = TASK_CONFIG[task_id]
    max_steps  = cfg["max_steps"]
    threshold  = cfg["threshold"]

    rewards:      List[float] = []
    history:      List[str]   = []   # action history fed back to LLM each step
    steps_taken:  int         = 0
    score:        float       = 0.0
    prev_score:   float       = 0.0
    success:      bool        = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result    = await env.reset(task_id=task_id)
        obs       = result.observation
        prev_score = obs.current_score

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action = get_model_action(client, obs, history)

            result = await env.step(action)
            obs    = result.observation

            reward      = result.reward or 0.0
            done        = result.done
            error       = obs.last_action_error if not obs.last_action_success else None
            score_delta = obs.current_score - prev_score
            prev_score  = obs.current_score

            rewards.append(reward)
            steps_taken = step

            # Build a rich history entry the LLM can learn from
            action_desc = _action_to_str(action)
            status      = "✓" if obs.last_action_success else "✗"
            delta_str   = f"+{score_delta:.4f}" if score_delta > 0 else f"{score_delta:.4f}"
            history.append(
                f"step={step} {status} {action_desc} reward={reward:+.2f} "
                f"score_delta={delta_str} score={obs.current_score:.4f}"
                + (f" ERROR={error}" if error else "")
            )

            log_step(
                step=step,
                action=action_desc,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        score   = obs.current_score if obs else 0.0
        success = score >= threshold

    finally:
        score   = score if score else 0.0
        success = success if success else False
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task":    task_id,
        "score":   score,
        "reward":  sum(rewards),
        "steps":   steps_taken,
        "success": success,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"API_BASE_URL     : {API_BASE_URL}")
    print(f"MODEL_NAME       : {MODEL_NAME}")
    print(f"LOCAL_IMAGE_NAME : {LOCAL_IMAGE_NAME}")
    print()

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if os.getenv("SPACE_ID"):   # Running inside HF Space
        env = DataCleaningEnv(base_url="http://localhost:7860")
        await env.connect()
    else:
        env = await DataCleaningEnv.from_docker_image(LOCAL_IMAGE_NAME)

    results = []
    try:
        for task_id in ("easy", "medium", "hard"):
            summary = await run_episode(env, client, task_id)
            results.append(summary)
            print()  # blank line between tasks
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    # ── Summary table ────────────────────────────────────────────────────────
    print("═" * 56)
    print(f"{'Task':<12} {'Score':>7}  {'Reward':>7}  {'Steps':>5}  {'Pass'}")
    print("─" * 56)
    for r in results:
        flag = "YES" if r["success"] else " NO"
        print(f"{r['task']:<12} {r['score']:>7.4f}  {r['reward']:>7.4f}  {r['steps']:>5}  {flag}")
    print("═" * 56)


if __name__ == "__main__":
    asyncio.run(main())