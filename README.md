 Meta PyTorch OpenEnv Hackathon x Scaler School of Technology


---
title: Tenant Negotiation Environment
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
---

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










