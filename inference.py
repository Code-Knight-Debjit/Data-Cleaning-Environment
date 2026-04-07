"""
inference.py
------------
Official submission inference script for the Data Cleaning Pipeline environment.

Reads from environment variables (ALL FREE — no paid API needed):
    API_BASE_URL       LLM endpoint. Default: HuggingFace free router.
    MODEL_NAME         Model to use.  Default: free open model.
    HF_TOKEN           Your free HuggingFace token (hf_...).
    LOCAL_IMAGE_NAME   Docker image name if using from_docker_image().
                       Leave unset to connect via ENV_BASE_URL instead.
    ENV_BASE_URL       Direct server URL. Default: http://localhost:8000

STDOUT FORMAT (evaluator parses these lines exactly — do not modify):
    [START] task=<n> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import re
import sys
from typing import List, Optional
from unittest import result
from client import DataCleaningEnvClient, CleanAction, CleanObservation
from openai import OpenAI

# ── Environment client imports ────────────────────────────────────────────────
try:
    from client import DataCleaningEnv
    from models import CleanAction, MAX_STEPS, DONE_THRESHOLD
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from client import DataCleaningEnv
    from models import CleanAction, MAX_STEPS, DONE_THRESHOLD


# ── Configuration — all defaults are FREE ────────────────────────────────────

API_BASE_URL     = os.getenv("API_BASE_URL",     "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",       "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN",         "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "openenv-data_cleaning:latest")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL",     "http://localhost:8000")

BENCHMARK  = "data_cleaning_env"
TASK_IDS   = ["easy", "medium", "hard"]

# Conservative budgets — keeps total runtime under 20 min on vcpu=2 / 8 GB
STEP_LIMITS = {"easy": 25, "medium": 50, "hard": 80}


# ── Official log helpers ──────────────────────────────────────────────────────
# Field names, order, and spacing match the evaluator spec exactly.

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step:   int,
    action: str,
    reward: float,
    done:   bool,
    error:  Optional[str],
) -> None:
    error_val  = error if error else "null"
    done_val   = str(done).lower()
    action_str = action[:80].replace("\n", " ")   # keep line single-line
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps:   int,
    score:   float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM helpers ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a data cleaning agent. You receive a dirty CSV and must fix it "
    "step by step using JSON action commands. Fix the most impactful issues "
    "first. Be precise — wrong column names cause errors. "
    "Output a single valid JSON object and nothing else — no explanation, no markdown."
)


def build_prompt(obs) -> str:
    rows      = obs.dirty_csv.strip().split("\n")
    preview   = "\n".join(rows[:30])
    truncated = len(rows) > 30
    last_err  = f"\nLast error: {obs.last_action_error}" if obs.last_action_error else ""
    return (
        f"Task: {obs.task_id}\n"
        f"Schema: {obs.schema_hint}\n"
        f"Score: {obs.current_score:.4f} | Issues remaining: {obs.issues_remaining}\n"
        f"Step {obs.step_number}/{obs.max_steps}{last_err}\n"
        f"\nCSV{' (first 30 rows)' if truncated else ''}:\n{preview}\n\n"
        "Reply with ONE JSON action:\n"
        '  {"command":"SET_VALUE",       "row_index":<int>, "column":"<name>", "value":"<str>"}\n'
        '  {"command":"DROP_ROW",        "row_index":<int>}\n'
        '  {"command":"STANDARDIZE_COL", "column":"<name>"}\n'
        '  {"command":"FILL_MISSING",    "column":"<name>", "fill_strategy":"mean|median|mode|drop"}\n'
        '  {"command":"DONE"}\n'
        "row_index = integer in the leftmost column of the CSV. JSON only."
    )


def parse_action(raw: str) -> CleanAction:
    """Convert model output to CleanAction. Falls back to DONE on any error."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        text  = "\n".join(inner).strip()
    try:
        return CleanAction(**json.loads(text))
    except Exception:
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if m:
            try:
                return CleanAction(**json.loads(m.group()))
            except Exception:
                pass
    return CleanAction(command="DONE")


def call_llm(client: OpenAI, messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=150,   # actions are short; saves free-tier quota
        temperature=0.1,
    )
    return (response.choices[0].message.content or "").strip()


# ── Episode loop ───────────────────────────────────────────────────────────────

async def run_episode(env, client: OpenAI, task_id: str) -> dict:
    """Run one episode. Emits [START] → N×[STEP] → [END]."""
    max_steps        = STEP_LIMITS[task_id]
    threshold        = DONE_THRESHOLD[task_id]
    rewards: List[float] = []
    steps_taken      = 0
    score            = 0.0
    success          = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs    = result.observation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            steps_taken = step
            messages.append({"role": "user", "content": build_prompt(obs)})

            try:
                raw    = call_llm(client, messages)
                action = parse_action(raw)
                messages.append({"role": "assistant", "content": raw})
            except Exception as exc:
                # API or parse failure — log and stop episode
                log_step(step, "DONE", 0.00, True, str(exc)[:120])
                rewards.append(0.0)
                break

            # Keep only system + last 8 exchanges to stay inside free-tier context limits
            if len(messages) > 17:
                messages = [messages[0]] + messages[-16:]

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

            if obs.done or score >= threshold:
                break

        success = score >= threshold

    finally:
        # [END] is always emitted, even if the episode crashed
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score,
            "reward": sum(rewards), "steps": steps_taken, "success": success}


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    if not HF_TOKEN:
        print(
            "ERROR: HF_TOKEN is not set.\n"
            "1. Go to https://huggingface.co/settings/tokens\n"
            "2. Click 'New token' → choose 'Read' → copy it\n"
            "3. In PowerShell: $env:HF_TOKEN='hf_xxxxxxxxxxxx'\n"
            "4. Then run: python inference.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"API_BASE_URL     : {API_BASE_URL}", flush=True)
    print(f"MODEL_NAME       : {MODEL_NAME}",   flush=True)
    print(f"LOCAL_IMAGE_NAME : {LOCAL_IMAGE_NAME or '(not set — using ENV_BASE_URL)'}", flush=True)
    print(f"ENV_BASE_URL     : {ENV_BASE_URL}",  flush=True)
    print("", flush=True)
    action = CleanAction(command="drop_column", column="some_col")
    result = await env.step(action)
    obs: CleanObservation = result.observation
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

    # Human-readable summary (evaluator ignores lines that don't start with [START]/[STEP]/[END])
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