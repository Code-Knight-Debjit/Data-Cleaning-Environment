"""
inference.py
------------
Official submission inference script for the Data Cleaning Pipeline environment.

Reads from environment variables (ALL FREE — no paid API needed):
  API_BASE_URL      LLM endpoint. Default: HuggingFace free router.
  MODEL_NAME        Model to use. Default: free open model.
  HF_TOKEN          Your free HuggingFace token (hf_...).
  LOCAL_IMAGE_NAME  Docker image name if using from_docker_image().
                    Leave unset to connect via ENV_BASE_URL instead.
  ENV_BASE_URL      Direct server URL. Default: http://localhost:8000

STDOUT FORMAT (evaluator parses these lines exactly — do not modify):
  [START] task=<n> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import asyncio
import io
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# ── Environment client imports ────────────────────────────────────────────────
try:
    from client import DataCleaningEnv
    from models import CleanAction, MAX_STEPS, DONE_THRESHOLD
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from client import DataCleaningEnv
    from models import CleanAction, MAX_STEPS, DONE_THRESHOLD

from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN         = os.getenv("HF_TOKEN", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL", "http://localhost:8000")

BENCHMARK   = "data_cleaning_env"
TASK_IDS    = ["easy", "medium", "hard"]
STEP_LIMITS = {"easy": 25, "medium": 50, "hard": 80}

# ── Official log helpers ──────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action[:80].replace(chr(10),' ')} "
          f"reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
          flush=True)


# =============================================================================
# COLUMN-LEVEL COMPLETION TRACKER
# Fixes: repeated transforms, brute-force row edits, no memory of done work.
# =============================================================================

class ColumnTracker:
    """
    Tracks per-column completion state to enforce masking and prevent
    repeated transformations.

    Per column:
      standardized      — STANDARDIZE_COL succeeded (score improved)
      filled            — FILL_MISSING succeeded (score improved)
      set_value_rows    — row indices already fixed via SET_VALUE
    """

    def __init__(self) -> None:
        self.standardized:       Set[str] = set()
        self.filled:             Set[str] = set()
        self.set_value_rows:     Dict[str, Set[int]] = {}
        self.consecutive_sv:     int  = 0
        self._prev_was_sv:       bool = False

    def record(self, action: CleanAction, score_delta: float) -> None:
        """Update completion state after a step."""
        cmd      = action.command
        improved = score_delta > 0.001

        if cmd == "STANDARDIZE_COL" and action.column and improved:
            self.standardized.add(action.column)
            self._prev_was_sv = False
            self.consecutive_sv = 0

        elif cmd == "FILL_MISSING" and action.column and improved:
            self.filled.add(action.column)
            self._prev_was_sv = False
            self.consecutive_sv = 0

        elif cmd == "SET_VALUE" and action.column and action.row_index is not None:
            col = action.column
            self.set_value_rows.setdefault(col, set()).add(action.row_index)
            self.consecutive_sv = (self.consecutive_sv + 1) if self._prev_was_sv else 1
            self._prev_was_sv = True

        else:
            self._prev_was_sv = False
            self.consecutive_sv = 0

    def is_blocked(self, cmd: str, column: Optional[str] = None,
                   row_index: Optional[int] = None) -> Tuple[bool, str]:
        """Return (blocked, reason). Soft masking — shown in prompt."""
        if cmd == "STANDARDIZE_COL" and column in self.standardized:
            return True, f"already standardized '{column}' with score gain"
        if cmd == "FILL_MISSING" and column in self.filled:
            return True, f"already filled '{column}' with score gain"
        if cmd == "SET_VALUE" and column and row_index is not None:
            if row_index in self.set_value_rows.get(column, set()):
                return True, f"row {row_index} col '{column}' already SET"
        return False, ""

    def brute_force_warning(self) -> Optional[str]:
        if self.consecutive_sv >= 3:
            return (
                f"STOP: {self.consecutive_sv} consecutive SET_VALUE calls — "
                f"BRUTE FORCE DETECTED. Use FILL_MISSING or STANDARDIZE_COL instead. "
                f"Row-by-row edits are inefficient and earn near-zero reward."
            )
        return None

    def ledger(self) -> str:
        lines = ["COMPLETION LEDGER (never redo completed work):"]
        if self.standardized:
            lines.append(f"  STANDARDIZED : {', '.join(sorted(self.standardized))}")
        if self.filled:
            lines.append(f"  FILLED       : {', '.join(sorted(self.filled))}")
        for col, rows in self.set_value_rows.items():
            lines.append(f"  SET_VALUE on '{col}': rows {sorted(rows)}")
        if len(lines) == 1:
            lines.append("  (nothing completed yet)")
        return "\n".join(lines)


# =============================================================================
# DATASET ANALYSIS
# =============================================================================

def _parse_csv(csv_str: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(io.StringIO(csv_str), index_col=0)
    except Exception:
        return None


def _analyse(df: pd.DataFrame) -> dict:
    """Compute column-level issue counts from the current DataFrame."""
    out: dict = {
        "rows": len(df),
        "cols": list(df.columns),
        "missing": {},
        "format_issues": {},
        "dup_rows": 0,
    }
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n:
            out["missing"][col] = n

    numeric_kw = {"price", "amount", "qty", "quantity", "value", "count"}
    for col in df.columns:
        series = df[col].dropna()
        if not len(series):
            continue
        if any(kw in col.lower() for kw in numeric_kw):
            coerced = pd.to_numeric(series, errors="coerce")
            n_bad   = int(coerced.isna().sum())
            if n_bad:
                examples = series[coerced.isna()].astype(str).unique()[:3].tolist()
                out["format_issues"][col] = {"count": n_bad, "examples": examples}

    out["dup_rows"] = int(df.duplicated().sum())
    return out


def _state_block(df: pd.DataFrame, obs) -> str:
    a = _analyse(df)
    L = [
        f"=== DATASET STATE  {a['rows']} rows x {len(a['cols'])} cols ===",
        f"SCORE={obs.current_score:.4f}  ISSUES_REMAINING={obs.issues_remaining}"
        f"  STEP={obs.step_number}/{obs.max_steps}",
    ]

    L.append("\nMISSING VALUES  (fix entire column with FILL_MISSING — 1 step):")
    if a["missing"]:
        for col, n in a["missing"].items():
            L.append(f"  {col:22s}  {n:3d} missing")
    else:
        L.append("  none")

    L.append("\nFORMAT ISSUES  (fix entire column with STANDARDIZE_COL — 1 step):")
    if a["format_issues"]:
        for col, info in a["format_issues"].items():
            L.append(f"  {col:22s}  {info['count']:3d} bad, e.g. {info['examples']}")
    else:
        L.append("  none")

    L.append(f"\nDUPLICATE ROWS: {a['dup_rows']}" if a["dup_rows"]
             else "\nDUPLICATE ROWS: none")

    L.append("\nDATA PREVIEW  (row_index = leftmost integer — use in SET_VALUE/DROP_ROW):")
    L.append(df.head(8).to_string(max_cols=10))
    return "\n".join(L)


def _action_menu(df: Optional[pd.DataFrame],
                 tracker: ColumnTracker, obs) -> str:
    """Build the allowed/blocked action list for the prompt."""
    if df is None:
        return "ACTIONS: FILL_MISSING | STANDARDIZE_COL | DROP_ROW | SET_VALUE | DONE"

    a        = _analyse(df)
    allowed  = []
    blocked  = []
    score    = obs.current_score

    for col, n in a["missing"].items():
        blk, reason = tracker.is_blocked("FILL_MISSING", col)
        if blk:
            blocked.append(f'FILL_MISSING  "{col}"  [SKIP: {reason}]')
        else:
            allowed.append(f'FILL_MISSING  col="{col}"  fill_strategy="median"   <- fixes {n} cells')

    for col, info in a["format_issues"].items():
        blk, reason = tracker.is_blocked("STANDARDIZE_COL", col)
        if blk:
            blocked.append(f'STANDARDIZE_COL  "{col}"  [SKIP: {reason}]')
        else:
            allowed.append(
                f'STANDARDIZE_COL  col="{col}"   <- fixes {info["count"]} format errors')

    if a["dup_rows"] > 0:
        dup_idx = df[df.duplicated(keep="first")].index.tolist()
        for idx in dup_idx[:3]:
            allowed.append(f'DROP_ROW  row_index={idx}   <- duplicate')

    if not a["missing"] and not a["format_issues"] and not a["dup_rows"]:
        allowed.append("SET_VALUE  row_index=<int>  column=<col>  value=<str>"
                       "   <- use for isolated single-cell anomalies only")

    if score >= 0.85 or obs.issues_remaining == 0:
        allowed.append(f"DONE   <- score={score:.3f} qualifies for completion")
    else:
        blocked.append(f"DONE   [BLOCKED: score={score:.3f} < 0.85, "
                       f"issues={obs.issues_remaining}]")

    lines = ["ALLOWED ACTIONS (pick the top one — sorted by impact):"]
    for a_ in allowed[:8]:
        lines.append(f"  OK  {a_}")
    if blocked:
        lines.append("BLOCKED (do NOT use):")
        for b in blocked[:6]:
            lines.append(f"  NO  {b}")
    return "\n".join(lines)


# =============================================================================
# TRAJECTORY MEMORY
# =============================================================================

def _trajectory(history: list) -> str:
    if not history:
        return "  (none yet)"
    lines = []
    for i, e in enumerate(history[-8:], 1):
        r     = f"{e['reward']:+.4f}" if e["reward"] is not None else "n/a"
        delta = f"  Dscore={e['score_delta']:+.4f}" if e.get("score_delta") is not None else ""
        err   = f"  ERROR: {e['error']}"            if e.get("error")        else ""
        lines.append(f"  {i}. {e['action_str']:48s} reward={r}{delta}{err}")
    return "\n".join(lines)


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """\
You are a cost-aware data cleaning optimizer.
Goal: fix a dirty CSV to match a target schema in the fewest steps possible.

MANDATORY REASONING (do this every turn):
  1. Read DATASET STATE   — identify column-level issues (missing, format, dups)
  2. Read COMPLETION LEDGER — confirm you are not repeating done work
  3. Read ALLOWED ACTIONS  — pick the action that fixes the MOST cells at once
  4. Output ONE JSON object — nothing else, no explanation, no markdown

EFFICIENCY LAWS (breaking these earns negative reward):
  * Column-level fix > row-level fix always.
    FILL_MISSING fixes ALL missing in a column in ONE step.
    SET_VALUE fixes ONE cell. Never use SET_VALUE when FILL_MISSING applies.
  * Never apply STANDARDIZE_COL or FILL_MISSING twice to the same column.
  * Never SET_VALUE a row you already SET.
  * Never call DONE if score < 0.85 and issues_remaining > 0.
  * 3+ consecutive SET_VALUE calls is brute-force. Stop and use column fix.

COMMANDS:
  {"command":"FILL_MISSING",    "column":"<col>", "fill_strategy":"mean|median|mode|drop"}
  {"command":"STANDARDIZE_COL", "column":"<col>"}
  {"command":"DROP_ROW",        "row_index":<int>}
  {"command":"SET_VALUE",       "row_index":<int>, "column":"<col>", "value":"<str>"}
  {"command":"DONE"}

Output: one JSON object, one line, nothing else."""


def _build_prompt(obs, df: Optional[pd.DataFrame],
                  tracker: ColumnTracker, history: list) -> str:
    parts = [
        f"TASK: {obs.task_id}",
        f"SCHEMA: {obs.schema_hint[:350]}",
    ]

    warn = tracker.brute_force_warning()
    if warn:
        parts.append(f"\n*** {warn} ***\n")

    if obs.last_action_error:
        parts.append(f"\nLAST ACTION FAILED: {obs.last_action_error}")
        parts.append("Choose a DIFFERENT action.\n")

    if df is not None:
        parts.append(_state_block(df, obs))
    else:
        rows = obs.dirty_csv.strip().split("\n")
        parts.append("CSV:\n" + "\n".join(rows[:12]))

    parts.append("")
    parts.append(tracker.ledger())
    parts.append("")
    parts.append(_action_menu(df, tracker, obs))
    parts.append("")
    parts.append("PREVIOUS STEPS (avoid repeating ineffective actions):")
    parts.append(_trajectory(history))
    parts.append("")
    parts.append("Your JSON action:")
    return "\n".join(parts)


# =============================================================================
# ACTION PARSING
# =============================================================================

def _parse_action(raw: str) -> CleanAction:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        text  = "\n".join(inner).strip()
    try:
        return CleanAction(**json.loads(text))
    except Exception:
        pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return CleanAction(**json.loads(m.group()))
        except Exception:
            pass
    return CleanAction(command="DONE")


def _action_str(action: CleanAction) -> str:
    parts = [action.command]
    if action.column:
        parts.append(f"col='{action.column}'")
    if action.row_index is not None:
        parts.append(f"row={action.row_index}")
    if action.value is not None:
        parts.append(f"val={action.value!r}")
    if action.fill_strategy:
        parts.append(f"strategy={action.fill_strategy}")
    return "  ".join(parts)


# =============================================================================
# LLM CALL
# =============================================================================

async def _call_llm(client: OpenAI, messages: list) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: (
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=200,
                temperature=0.05,
            ).choices[0].message.content or ""
        ).strip(),
    )


# =============================================================================
# EPISODE LOOP
# =============================================================================

async def run_episode(env, client: OpenAI, task_id: str) -> dict:
    max_steps  = STEP_LIMITS[task_id]
    threshold  = DONE_THRESHOLD[task_id]
    rewards:   List[float] = []
    steps_taken = 0
    score       = 0.0
    prev_score  = 0.0
    success     = False
    history:   list = []
    tracker    = ColumnTracker()

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result     = await env.reset(task_id=task_id)
        obs        = result.observation
        score      = obs.current_score
        prev_score = score

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            steps_taken = step
            df          = _parse_csv(obs.dirty_csv)

            user_msg = _build_prompt(obs, df, tracker, history)
            messages.append({"role": "user", "content": user_msg})

            try:
                raw    = await _call_llm(client, messages)
                action = _parse_action(raw)
                messages.append({"role": "assistant", "content": raw})
            except Exception as exc:
                log_step(step, "DONE", 0.00, True, str(exc)[:120])
                rewards.append(0.0)
                break

            # Rolling context window: system + last 6 exchanges
            if len(messages) > 13:
                messages = [messages[0]] + messages[-12:]

            result      = await env.step(action)
            obs         = result.observation
            reward      = result.reward or 0.0
            score       = obs.current_score
            score_delta = score - prev_score
            prev_score  = score
            rewards.append(reward)

            astr = _action_str(action)
            tracker.record(action, score_delta)
            history.append({
                "action_str":  astr,
                "reward":      reward,
                "score_delta": score_delta,
                "error":       obs.last_action_error,
            })

            log_step(step=step, action=astr, reward=reward,
                     done=obs.done, error=obs.last_action_error)

            if obs.done or score >= threshold:
                break

        success = score >= threshold

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score,
            "reward": sum(rewards), "steps": steps_taken, "success": success}


# =============================================================================
# ENTRY POINT
# =============================================================================

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
    print(f"MODEL_NAME       : {MODEL_NAME}", flush=True)
    print(f"LOCAL_IMAGE_NAME : {LOCAL_IMAGE_NAME or '(not set — using ENV_BASE_URL)'}", flush=True)
    print(f"ENV_BASE_URL     : {ENV_BASE_URL}", flush=True)
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

    print("=" * 58, flush=True)
    print(f"{'Task':<10} {'Score':>7} {'Reward':>9} {'Steps':>6} {'Pass':>5}")
    print("-" * 58, flush=True)
    for r in results:
        print(
            f"{r['task_id']:<10} {r['score']:>7.4f} {r['reward']:>9.4f} "
            f"{r['steps']:>6} {'YES' if r['success'] else 'NO':>4}",
            flush=True,
        )
    print("=" * 58, flush=True)


if __name__ == "__main__":
    asyncio.run(main())