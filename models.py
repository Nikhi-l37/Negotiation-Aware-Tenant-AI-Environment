"""
Pydantic type definitions for the Negotiation-Aware Tenant AI Environment.

Defines the Action, Observation, and State models that form the typed interface
between the agent and the environment, following the OpenEnv specification.
"""

from typing import List, Optional
from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Action — what the agent sends each step
# ---------------------------------------------------------------------------

class TenantAction(Action):
    """
    The agent chooses one or more boolean flags each month.
    At least one flag should be True for a meaningful step.
    """
    increase_rent: bool = False      # +10% rent, −0.15 trust
    offer_discount: bool = False     # −5% rent as cost, +0.20 trust
    perform_maintenance: bool = False  # −$100, +0.10 trust
    negotiate: bool = False          # +2% rent, +0.05 trust


# ---------------------------------------------------------------------------
# Observation — what the environment returns each step
# ---------------------------------------------------------------------------

class TenantObservation(Observation):
    """
    The observation returned after each step.
    Inherits `done: bool` and `reward: Optional[float]` from Observation base.
    """
    rent: float = 0.0                   # Current monthly rent ($)
    trust_score: float = 1.0            # Tenant trust, 0.0 – 1.0
    tenant_type: str = "loyal"          # "loyal", "price_sensitive", or "demanding"
    months_stayed: int = 0              # Steps elapsed in current episode
    is_vacant: bool = False             # True if tenant left
    market_rate: float = 0.0            # Current market rent for reference
    maintenance_due: bool = False       # Whether maintenance is expected this step
    message: str = ""                   # Human-readable status message


# ---------------------------------------------------------------------------
# State — internal episode metadata
# ---------------------------------------------------------------------------

class TenantState(State):
    """
    Extended state metadata for task-routing and grading.
    Inherits `episode_id: Optional[str]` and `step_count: int` from State base.
    """
    task_name: str = "easy"
    total_profit: float = 0.0
    total_cost: float = 0.0
    max_steps: int = 12
