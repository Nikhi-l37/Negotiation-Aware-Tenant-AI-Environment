# Negotiation-Aware Tenant AI Environment

An OpenEnv-compliant RL environment where an LLM agent manages a rental
property — negotiating rent, handling maintenance, and balancing short-term
profit against long-term tenant trust and retention.

Built for the **Meta OpenEnv Hackathon** using the
[`openenv-core`](https://github.com/meta-pytorch/OpenEnv) framework.

---

## Environment Description & Motivation

Property management is a real-world task involving ongoing strategic
decisions: when to raise rent, when to invest in maintenance, and how to
handle different tenant personalities.  This environment captures those
dynamics in a 12-month (12-step) episode with deterministic mechanics.

Unlike static simulations, this environment features **Dynamic Market Events**
— deterministic monthly shocks (pipe bursts, market booms, tenant job loss)
that force the agent to read the situation and adapt its strategy mid-episode.
An agent cannot simply repeat the same actions every month.

**Key design choices:**
- Realistic dollar values for rent ($1,000 – $3,000 range)
- Three distinct tenant archetypes with different failure modes
- Dynamic market events that test adaptive reasoning
- Multi-objective reward signal (not binary)
- Fully deterministic grading — no LLM-as-judge

---

## Action Space

| Action | Effect | Trust Impact |
|---|---|---|
| `increase_rent` | Rent ×1.10 | −0.15 |
| `offer_discount` | Cost = 5% of rent | +0.20 |
| `perform_maintenance` | Cost = $100 | +0.10 |
| `negotiate` | Rent ×1.02 | +0.05 |

All four are boolean flags — the agent can combine multiple actions per step.

## Observation Space

| Field | Type | Description |
|---|---|---|
| `rent` | float | Current monthly rent ($) |
| `trust_score` | float | Tenant trust, 0.0 – 1.0 |
| `tenant_type` | str | `"loyal"`, `"price_sensitive"`, or `"demanding"` |
| `months_stayed` | int | Steps elapsed (0 – 12) |
| `is_vacant` | bool | True if tenant left |
| `market_rate` | float | Market rent reference ($) |
| `maintenance_due` | bool | Whether maintenance is expected |
| `message` | str | Status message |

## Reward Shaping

Each step returns: `rent - cost` (positive if rent > costs).
- **Vacancy penalty**: −$5,000 if the tenant leaves
- **No-action penalty**: −0.03 trust from neglect
- Tenant-type modifiers amplify or dampen effects

---

## Dynamic Market Events

Deterministic monthly events inject real-world unpredictability into every episode.
The agent must read the `message` field and adapt — no single strategy works for
all 12 months.

| Month | Event | Effect | Agent Must... |
|---|---|---|---|
| **3** | 🔧 Pipe Burst | −0.40 trust if maintenance not performed | Do maintenance or face trust collapse |
| **7** | 📈 Market Boom | `market_rate` × 1.20 | Adapt pricing to the new market |
| **10** | 💼 Tenant Job Loss | Trust → 0 if rent increased | NOT raise rent or tenant leaves instantly |

These events are **fully deterministic** (tied to fixed months) so grading
remains reproducible. They communicate through the existing `message` field —
no schema changes required.

---

## Tasks & Difficulty

### Easy — Loyal Tenant
- **Tenant**: Loyal (trust changes halved)
- **Start**: $1,500 rent, $1,600 market, trust 1.0
- **Challenge**: Maximise rent while handling the pipe burst at month 3 and job loss at month 10

### Medium — Price-Sensitive Tenant
- **Tenant**: Price-sensitive (leaves if rent > 110% market)
- **Start**: $1,200 rent, $1,300 market, trust 0.8
- **Challenge**: Walk the fine line between profit and triggering departure; the market boom at month 7 shifts the ceiling

### Hard — Demanding Tenant
- **Tenant**: Demanding (expects maintenance every 3 months)
- **Start**: $2,000 rent, $2,200 market, trust 0.7
- **Challenge**: Mandatory maintenance at months 3, 6, 9, 12 — plus pipe burst overlaps with month 3 maintenance, and job loss at month 10 punishes greed

---

## Setup & Usage

### Prerequisites
```bash
pip install openenv-core pydantic openai fastapi uvicorn websockets
```

### Run Locally
```bash
# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, interact via the client
python -c "
from client import TenantEnv
from models import TenantAction
with TenantEnv(base_url='http://localhost:8000').sync() as env:
    result = env.reset(task_name='easy')
    print(result.observation)
    result = env.step(TenantAction(negotiate=True))
    print(result.observation)
"
```

### Run with Docker
```bash
docker build -t tenant-negotiation-env .
docker run -p 8000:8000 tenant-negotiation-env
```

### Run Inference (Baseline Agent)
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="sk-..."
python inference.py
```

### Validate
```bash
openenv validate
```

---

## Project Structure

```
.
├── models.py               ← Typed Action/Observation/State models
├── client.py               ← WebSocket client (what agents import)
├── server/
│   ├── environment.py      ← Core game logic (reset/step/state)
│   ├── app.py              ← FastAPI server (create_app wrapper)
│   └── __init__.py
├── tasks.py                ← 3 tasks with deterministic graders
├── inference.py            ← Baseline LLM agent script
├── openenv.yaml            ← OpenEnv manifest
├── requirements.txt        ← Python dependencies
├── Dockerfile              ← Container definition
└── README.md               ← This file
```

---

## Baseline Scores

| Task | Expected Range | Notes |
|---|---|---|
| Easy | 0.4 – 0.8 | Loyal tenant is forgiving, but pipe burst + job loss add risk |
| Medium | 0.3 – 0.7 | Market boom at month 7 changes the pricing ceiling |
| Hard | 0.2 – 0.5 | Mandatory maintenance + dynamic events make this genuinely hard |

_Scores depend on the LLM model used. GPT-4o achieves the upper ranges. Dynamic market events
mean the agent must read and react to the `message` field — static strategies will underperform._











PROBLEM STATEMENT

Round 1 — Problem Statement



THE TASK

Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard step() / reset() / state() API.



KEY REQUIREMENTS AT A GLANCE



Must simulate a real-world task (not games or toys)



Implement full OpenEnv spec: typed models, step()/reset()/state(), openenv.yaml



Minimum 3 tasks with agent graders (easy → medium → hard, scores/reward 0.0–1.0)



Meaningful reward function with partial progress signals



Baseline inference script with reproducible scores



Deploy to Hugging Face Spaces + working Dockerfile



README with environment description, action/observation spaces, setup instructions



Detailed Requirements

FUNCTIONAL REQUIREMENTS

Real-world task simulation

The environment must simulate a task humans actually do. Not games, not toys. Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.

OpenEnv spec compliance

Implement the full OpenEnv interface: typed Observation, Action, and Reward Pydantic models. step(action) → returns observation, reward, done, info. reset() → returns initial observation. state() → returns current state. openenv.yaml with metadata. Tested via openenv validate.

Minimum 3 tasks with agent graders

Each task defines a concrete objective an agent must accomplish, with a programmatic grader that scores performance (0.0–1.0). Tasks should range: easy → medium → hard. Graders must have clear, deterministic success/failure criteria.

Meaningful reward function

Provides signal over the full trajectory (not just binary end-of-episode). Rewards partial progress toward task completion. Penalizes clearly undesirable behavior (e.g. infinite loops, destructive actions).

Baseline inference script

Uses the OpenAI API client to run a model against the environment. Reads API credentials from environment variables (OPENAI_API_KEY). Produces a reproducible baseline score on all 3 tasks.









Evaluation Criteria

PARAMETER

WEIGHT

DESCRIPTION





Real-world utility

30%

Does the environment model a genuine task? Would someone actually use this to train or evaluate agents?





Task & grader quality

25%

Are tasks well-defined with clear objectives? Do graders accurately and fairly measure success? Meaningful difficulty progression?





Environment design

20%

Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries.





Code quality & spec compliance

15%

Follows OpenEnv spec, clean project structure, typed models, documented, tested, Dockerfile works.





Creativity & novelty

10%

Novel problem domain, interesting mechanics, clever reward design, original approach.









Real-world utility (30%)

• 0–5: Toy/artificial problem with no practical application

• 6–15: Valid domain but shallow modeling of the real task

• 16–25: Good domain modeling, would be useful for agent evaluation

• 26–30: Excellent — fills a real gap, immediate value for the RL/agent community

Task & grader quality (25%)

• 3+ tasks with difficulty range?

• Graders produce scores between 0.0–1.0?

• Graders deterministic and reproducible?

• Hard task genuinely challenges frontier models?

Environment design (20%)

• reset() produces clean state?

• Action/observation types well-designed and documented?

• Reward function provides useful varying signal (not just sparse)?

• Episode boundaries sensible?

Code quality & spec compliance (15%)

• openenv validate passes?

• docker build && docker run works?

• HF Space deploys and responds?

• Baseline script runs and reproduces scores?

Creativity & novelty (10%)

• Domain we haven’t seen in OpenEnv before?

• Reward design has interesting properties?

• Clever mechanics that make the environment engaging?









How Judging works



Phase 1: Automated Validation

Pass/fail gate — HF Space deploys, OpenEnv spec compliance, Dockerfile builds, baseline reproduces, 3+ tasks with graders.

Phase 2: Agentic Evaluation

Scored — baseline agent re-run, standard Open LLM agent (e.g. Nemotron 3 Super) run against all environments, score variance check.

Phase 3: Human Review

Top submissions reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and exploit checks.

DISQUALIFICATION CRITERIA



Environment does not deploy or respond

Plagiarized or trivially modified existing environments

Graders that always return the same score

No baseline inference script









Pre-Submission Checklist — all must pass or you're disqualified

HF Space deploys

Automated ping to the Space URL — must return 200 and respond to reset()





OpenEnv spec compliance

Validate openenv.yaml, typed models, step()/reset()/state() endpoints





Dockerfile builds

Automated docker build on the submitted repo





Baseline reproduces

Run the submitted inference script — must complete without error and produce scores





3+ tasks with graders

Enumerate tasks, run each grader, verify scores/reward in 0.0–1.0 range





Mandatory Additional Instructions

Before submitting, ensure the following variables are defined in your environment configuration:

API_BASE_URL The API endpoint for the LLM.

MODEL_NAME The model identifier to use for inference.

HF_TOKEN Your Hugging Face / API key.

The inference script must be named `inference.py` and placed in the root directory of the project

Participants must use OpenAI Client for all LLM calls using above variables

Participants must emit structured stdout logs strictly following the [START], [STEP], and [END] format defined in the sample inference.py provided below. Any deviation in field names, ordering, or formatting will result in incorrect evaluation scoring. Refer to the Sample Inference Script for the complete format specification and examples.





Infra Restrictions

Runtime of inference script should be less than 20min

Make sure your env and inference can run on a machine with vcpu=2, memory=8gb





Validator

Run the pre-submission validation script before submitting











Sample Inference Script



#!/usr/bin/env bash

#

# validate-submission.sh — OpenEnv Submission Validator

#

# Checks that your HF Space is live, Docker image builds, and openenv validate passes.

#

# Prerequisites:

#   - Docker:       https://docs.docker.com/get-docker/

#   - openenv-core: pip install openenv-core

#   - curl (usually pre-installed)

#

# Run:

#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]

#

#   Or download and run locally:

#     chmod +x validate-submission.sh

#     ./validate-submission.sh <ping_url> [repo_dir]

#

# Arguments:

#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)

#   repo_dir   Path to your repo (default: current directory)

#

# Examples:

#   ./validate-submission.sh https://my-team.hf.space

#   ./validate-submission.sh https://my-team.hf.space ./my-repo

#



set -uo pipefail



DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then

  RED='\033[0;31m'

  GREEN='\033[0;32m'

  YELLOW='\033[1;33m'

  BOLD='\033[1m'

  NC='\033[0m'

else

  RED='' GREEN='' YELLOW='' BOLD='' NC=''

fi



run_with_timeout() {

  local secs="$1"; shift

  if command -v timeout &>/dev/null; then

    timeout "$secs" "$@"

  elif command -v gtimeout &>/dev/null; then

    gtimeout "$secs" "$@"

  else

    "$@" &

    local pid=$!

    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &

    local watcher=$!

    wait "$pid" 2>/dev/null

    local rc=$?

    kill "$watcher" 2>/dev/null

    wait "$watcher" 2>/dev/null

    return $rc

  fi

}



portable_mktemp() {

  local prefix="${1:-validate}"

  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp

}



CLEANUP_FILES=()

cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }

trap cleanup EXIT



PING_URL="${1:-}"

REPO_DIR="${2:-.}"



if [ -z "$PING_URL" ]; then

  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"

  printf "\n"

  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"

  printf "  repo_dir   Path to your repo (default: current directory)\n"

  exit 1

fi



if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then

  printf "Error: directory '%s' not found\n" "${2:-.}"

  exit 1

fi

PING_URL="${PING_URL%/}"

export PING_URL

PASS=0



log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }

pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }

fail() { log "${RED}FAILED${NC} -- $1"; }

hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }

stop_at() {

  printf "\n"

  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"

  exit 1

}



printf "\n"

printf "${BOLD}========================================${NC}\n"

printf "${BOLD}  OpenEnv Submission Validator${NC}\n"

printf "${BOLD}========================================${NC}\n"

log "Repo:     $REPO_DIR"

log "Ping URL: $PING_URL"

printf "\n"



log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."



CURL_OUTPUT=$(portable_mktemp "validate-curl")

CLEANUP_FILES+=("$CURL_OUTPUT")

HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \

  -H "Content-Type: application/json" -d '{}' \

  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")



if [ "$HTTP_CODE" = "200" ]; then

  pass "HF Space is live and responds to /reset"

elif [ "$HTTP_CODE" = "000" ]; then

  fail "HF Space not reachable (connection failed or timed out)"

  hint "Check your network connection and that the Space is running."

  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"

  stop_at "Step 1"

else

  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"

  hint "Make sure your Space is running and the URL is correct."

  hint "Try opening $PING_URL in your browser first."

  stop_at "Step 1"

fi



log "${BOLD}Step 2/3: Running docker build${NC} ..."



if ! command -v docker &>/dev/null; then

  fail "docker command not found"

  hint "Install Docker: https://docs.docker.com/get-docker/"

  stop_at "Step 2"

fi



if [ -f "$REPO_DIR/Dockerfile" ]; then

  DOCKER_CONTEXT="$REPO_DIR"

elif [ -f "$REPO_DIR/server/Dockerfile" ]; then

  DOCKER_CONTEXT="$REPO_DIR/server"

else

  fail "No Dockerfile found in repo root or server/ directory"

  stop_at "Step 2"

fi



log "  Found Dockerfile in $DOCKER_CONTEXT"



BUILD_OK=false

BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true



if [ "$BUILD_OK" = true ]; then

  pass "Docker build succeeded"

else

  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"

  printf "%s\n" "$BUILD_OUTPUT" | tail -20

  stop_at "Step 2"

fi



log "${BOLD}Step 3/3: Running openenv validate${NC} ..."



if ! command -v openenv &>/dev/null; then

  fail "openenv command not found"

  hint "Install it: pip install openenv-core"

  stop_at "Step 3"

fi



VALIDATE_OK=false

VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true



if [ "$VALIDATE_OK" = true ]; then

  pass "openenv validate passed"

  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"

else

  fail "openenv validate failed"

  printf "%s\n" "$VALIDATE_OUTPUT"

  stop_at "Step 3"

fi



printf "\n"

printf "${BOLD}========================================${NC}\n"

printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"

printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"

printf "${BOLD}========================================${NC}\n"

printf "\n"



exit 0









"""

Inference Script Example

===================================

MANDATORY

- Before submitting, ensure the following variables are defined in your environment configuration:

    API_BASE_URL   The API endpoint for the LLM.

    MODEL_NAME     The model identifier to use for inference.

    HF_TOKEN       Your Hugging Face / API key.

    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

                     method



- Defaults are set only for API_BASE_URL and MODEL_NAME 

    (and should reflect your active inference setup):

    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")

    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

    

- The inference script must be named `inference.py` and placed in the root directory of the project

- Participants must use OpenAI Client for all LLM calls using above variables



STDOUT FORMAT

- The script must emit exactly three line types to stdout, in this order:



    [START] task=<task_name> env=<benchmark> model=<model_name>

    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>

    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>



  Rules:

    - One [START] line at episode begin.

    - One [STEP] line per step, immediately after env.step() returns.

    - One [END] line after env.close(), always emitted (even on exception).

    - reward and rewards are formatted to 2 decimal places.

    - done and success are lowercase booleans: true or false.

    - error is the raw last_action_error string, or null if none.

    - All fields on a single line with no newlines within a line.

    - Each tasks should return score in [0, 1]



  Example:

    [START] task=click-test env=miniwob model=Qwen3-VL-30B

    [STEP] step=1 action=click('123') reward=0.00 done=false error=null

    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null

    [STEP] step=3 action=click('789') reward=1.00 done=true error=null

    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00

"""



import asyncio

import os

import textwrap

from typing import List, Optional



from openai import OpenAI



from my_env_v4 import MyEnvV4Action, MyEnvV4Env

IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")



API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"

MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")

BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")

MAX_STEPS = 8

TEMPERATURE = 0.7

MAX_TOKENS = 150

SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]



# Max possible reward: each token contributes 0.1, across all steps

_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1

MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP



SYSTEM_PROMPT = textwrap.dedent(

    """

    You are interacting with a simple echo environment.

    Each turn you must send a message. The environment will echo it back.

    Reward is proportional to message length: reward = len(message) * 0.1

    Your goal is to maximize total reward by sending meaningful, substantive messages.

    Reply with exactly one message string — no quotes, no prefixes, just the message text.

    """

).strip()





def log_start(task: str, env: str, model: str) -> None:

    print(f"[START] task={task} env={env} model={model}", flush=True)





def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:

    error_val = error if error else "null"

    done_val = str(done).lower()

    print(

        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",

        flush=True,

    )





def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)





def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:

    history_block = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(

        f"""

        Step: {step}

        Last echoed message: {last_echoed!r}

        Last reward: {last_reward:.2f}

        Previous steps:

        {history_block}

        Send your next message.

        """

    ).strip()





def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:

    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)

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

        return text if text else "hello"

    except Exception as exc:

        print(f"[DEBUG] Model request failed: {exc}", flush=True)

        return "hello"





async def main() -> None:

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)



    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)



    history: List[str] = []

    rewards: List[float] = []

    steps_taken = 0

    score = 0.0

    success = False



    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)



    try:

        result = await env.reset() # OpenENV.reset()

        last_echoed = result.observation.echoed_message

        last_reward = 0.0



        for step in range(1, MAX_STEPS + 1):

            if result.done:

                break



            message = get_model_message(client, step, last_echoed, last_reward, history)



            result = await env.step(MyEnvV4Action(message=message))

            obs = result.observation



            reward = result.reward or 0.0

            done = result.done

            error = None



            rewards.append(reward)

            steps_taken = step

            last_echoed = obs.echoed_message

            last_reward = reward



            log_step(step=step, action=message, reward=reward, done=done, error=error)



            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")



            if done:

                break



        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0

        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]

        success = score >= SUCCESS_SCORE_THRESHOLD



    finally:

        try:

            await env.close()

        except Exception as e:

            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)





if __name__ == "__main__":

    asyncio.run(main())















When Round 1 opens, you'll choose 1 of 4–5 problem statements and build an OpenEnv environment around it.

Example of what a problem statement looks like

"Build a mini-game RL environment with clearly defined tasks, automated graders, and reward logic using the OpenEnv framework."

→ Create a mini-game an AI agent can play

→ Define tasks with increasing difficulty

→ Write graders that verify task completion

→ Define reward logic for scoring

→ Package using OpenEnv for automated evaluation

Evaluation Criteria

Runtime correctness

Runs without errors





Interface compliance

Follows OpenEnv standard





Task design

Clear, realistic, testable



Grading logic

Reward system makes sense









Install before April 1st.

Python 3.10+

Install 3.10, 3.11, or 3.12.

$

Copy

Git + GitHub account

Push your submission to GitHub or HF.

$

Copy

Hugging Face CLI

Deploy to HF Spaces.

$

Copy

$

Copy

OpenEnv

The framework.

$

Copy

Google Colab

Prep course runs in Colab. Free tier works.

$

Copy

OpenEnv

The framework.

Copy

Docker

Isolated container testing.

Copy

Recommended

VS Code

Best Python + Docker support











When Round 1 starts on 1 April:



STEP 1

Application Form

Choose 1 of the 4–5 problem statements revealed on the platform.



STEP 2

Scaffold

$

Copy

Generate project structure.



STEP 3

Build

Define your environment in the generated files.



STEP 4

Test locally

$

Copy



STEP 5

Deploy

$

Copy

STEP 6

Submit

Paste your HF Spaces URL here before the deadline.



Deadline: 8 April 2026, 11:59 PM IST









these is the detail requirments of the hackathon