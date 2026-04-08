"""
Baseline inference script for the Negotiation-Aware Tenant AI Environment.

Reads config from environment variables, connects to the deployed HF Space,
runs an LLM agent against each task, and emits structured [START]/[STEP]/[END]
logs required by the Meta OpenEnv evaluation pipeline.

Environment variables required:
    API_BASE_URL   — LLM endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME     — model identifier (e.g. gpt-4o)
    HF_TOKEN       — Hugging Face token (also used as OPENAI_API_KEY)
"""

import asyncio
import json
import os
import sys
from typing import List

from openai import OpenAI

# ── Local imports ──────────────────────────────────────────────────────
from client import TenantEnv
from models import TenantAction

# ── Configuration ─────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
IMAGE_NAME = os.getenv("IMAGE_NAME", "tenant-negotiation-env")

TASK_NAMES = ["easy", "medium", "hard"]
MAX_STEPS = 12
BENCHMARK = "tenant-negotiation-env"
TEMPERATURE = 0.2
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# Maximum total reward for normalization (rough upper bound)
MAX_TOTAL_REWARD = 22000.0

# ── System prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert AI property manager. Each month you receive an observation
about your rental property and tenant, then you must choose actions.

Available actions (all booleans, combine as needed):
- increase_rent: Raise rent by 10%. Boosts profit but hurts trust (−0.15).
- offer_discount: Give a 5% discount. Costs money but boosts trust (+0.20).
- perform_maintenance: Spend $100 on repairs. Boosts trust (+0.10).
                       CRITICAL for "demanding" tenants every 3 months.
- negotiate: Talk it out. Small rent increase (+2%) and trust boost (+0.05).

Rules:
- If trust_score drops below 0.2, the tenant LEAVES (huge penalty).
- "price_sensitive" tenants leave if rent exceeds 110% of market_rate.
- "demanding" tenants expect maintenance every 3 months (maintenance_due=true).
- "loyal" tenants are forgiving; trust changes are halved.

Goal: Maximize total profit over 12 months while keeping the tenant.

Respond with ONLY a JSON object like:
{"increase_rent": false, "offer_discount": false, "perform_maintenance": true, "negotiate": true}
"""


# ── Structured logging (matches Meta OpenEnv validator format) ────────

def log_start(task: str, env: str, model: str) -> None:
    """Emit the [START] log block."""
    print(json.dumps({
        "type": "[START]",
        "task": task,
        "env": env,
        "model": model,
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    """Emit the [STEP] log block."""
    entry = {
        "type": "[STEP]",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
    }
    if error:
        entry["error"] = str(error)
    print(json.dumps(entry), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit the [END] log block."""
    print(json.dumps({
        "type": "[END]",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
    }), flush=True)


# ── LLM helper ────────────────────────────────────────────────────────

def get_model_message(
    client: OpenAI,
    step: int,
    observation: dict,
    last_reward: float,
    history: List[str],
) -> str:
    """Query the LLM for the next action JSON string."""
    user_prompt = (
        f"Month {step}/12.\n"
        f"Observation: {json.dumps(observation)}\n"
        f"Last reward: {last_reward:+.2f}\n"
        f"History summary: {'; '.join(history[-3:]) if history else 'none'}\n\n"
        f"Choose your actions as a JSON object."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "{}"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return "{}"


def parse_action(raw: str) -> TenantAction:
    """Parse the LLM's raw JSON string into a TenantAction, with safe fallback."""
    try:
        # Strip markdown fences if the model wraps in ```json ... ```
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0]
        data = json.loads(cleaned)
        return TenantAction(**data)
    except Exception:
        return TenantAction(negotiate=True)  # Safe default


# ── Main inference loop ───────────────────────────────────────────────

async def run_task(task_name: str) -> float:
    """Run a single task against the environment and return the final score."""
    client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
    )
    env = await TenantEnv.from_docker_image(IMAGE_NAME)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Build observation dict for the LLM
            obs_dict = {
                "rent": obs.rent,
                "trust_score": obs.trust_score,
                "tenant_type": obs.tenant_type,
                "months_stayed": obs.months_stayed,
                "is_vacant": obs.is_vacant,
                "market_rate": obs.market_rate,
                "maintenance_due": obs.maintenance_due,
                "message": obs.message,
            }

            raw_response = get_model_message(client, step, obs_dict, last_reward, history)
            action = parse_action(raw_response)

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            action_str = json.dumps({
                "increase_rent": action.increase_rent,
                "offer_discount": action.offer_discount,
                "perform_maintenance": action.perform_maintenance,
                "negotiate": action.negotiate,
            })

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    """Run inference across all three tasks."""
    for task_name in TASK_NAMES:
        await run_task(task_name)


if __name__ == "__main__":
    asyncio.run(main())
