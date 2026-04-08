"""
Core environment logic for the Negotiation-Aware Tenant AI Environment.

Implements the OpenEnv Environment base class with reset(), step(), and state property.
Manages tenant behaviour, trust dynamics, rent adjustments, and episode boundaries.
"""

import uuid
from openenv.core.env_server import Environment
from models import TenantAction, TenantObservation, TenantState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS = 12          # Each episode = 12 months
VACANCY_PENALTY = 5000  # Massive penalty when tenant leaves
MAINTENANCE_COST = 100  # Flat cost for performing maintenance
TRUST_FLOOR = 0.2       # If trust drops below this, tenant leaves


# ---------------------------------------------------------------------------
# Task configurations (deterministic initial conditions)
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "easy": {
        "tenant_type": "loyal",
        "rent": 1500.0,
        "market_rate": 1600.0,
        "trust_score": 1.0,
    },
    "medium": {
        "tenant_type": "price_sensitive",
        "rent": 1200.0,
        "market_rate": 1300.0,
        "trust_score": 0.8,
    },
    "hard": {
        "tenant_type": "demanding",
        "rent": 2000.0,
        "market_rate": 2200.0,
        "trust_score": 0.7,
    },
}


class TenantNegotiationEnvironment(Environment):
    """
    An RL environment where an agent manages a rental property by choosing
    rent adjustments, discounts, maintenance, or negotiation each month.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = TenantState()
        self._rent = 0.0
        self._market_rate = 0.0
        self._trust_score = 1.0
        self._tenant_type = "loyal"
        self._is_vacant = False
        self._total_profit = 0.0
        self._total_cost = 0.0
        self._step_rewards: list[float] = []

    # ------------------------------------------------------------------
    # OpenEnv interface: reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, episode_id=None, task_name="easy", **kwargs) -> TenantObservation:
        """Reset the environment to the initial state for the given task."""
        config = TASK_CONFIGS.get(task_name, TASK_CONFIGS["easy"])

        self._tenant_type = config["tenant_type"]
        self._rent = config["rent"]
        self._market_rate = config["market_rate"]
        self._trust_score = config["trust_score"]
        self._is_vacant = False
        self._total_profit = 0.0
        self._total_cost = 0.0
        self._step_rewards = []

        self._state = TenantState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            total_profit=0.0,
            total_cost=0.0,
            max_steps=MAX_STEPS,
        )

        maintenance_due = self._is_maintenance_due(1)
        return TenantObservation(
            done=False,
            reward=None,
            rent=self._rent,
            trust_score=self._trust_score,
            tenant_type=self._tenant_type,
            months_stayed=0,
            is_vacant=False,
            market_rate=self._market_rate,
            maintenance_due=maintenance_due,
            message=f"New episode started. Tenant type: {self._tenant_type}. "
                    f"Starting rent: ${self._rent:.0f}/mo.",
        )

    # ------------------------------------------------------------------
    # OpenEnv interface: step
    # ------------------------------------------------------------------

    def step(self, action: TenantAction, timeout_s=None, **kwargs) -> TenantObservation:
        """Execute one month of the simulation."""

        # If episode already over, return terminal observation
        if self._is_vacant or self._state.step_count >= MAX_STEPS:
            return self._terminal_observation("Episode already ended.")

        self._state.step_count += 1
        month = self._state.step_count

        # ---- Apply action effects ----
        cost = 0.0
        trust_delta = 0.0
        messages = []

        if action.increase_rent:
            self._rent *= 1.10
            trust_delta -= 0.15
            messages.append(f"Rent increased to ${self._rent:.0f}.")

        if action.offer_discount:
            discount_amount = self._rent * 0.05
            cost += discount_amount
            trust_delta += 0.20
            messages.append(f"Discount of ${discount_amount:.0f} offered. Trust improved.")

        if action.perform_maintenance:
            cost += MAINTENANCE_COST
            trust_delta += 0.10
            messages.append("Maintenance performed.")

        if action.negotiate:
            self._rent *= 1.02
            trust_delta += 0.05
            messages.append(f"Negotiated rent to ${self._rent:.0f}. Trust slightly improved.")

        # No action at all — slight trust decay from neglect
        if not any([action.increase_rent, action.offer_discount,
                     action.perform_maintenance, action.negotiate]):
            trust_delta -= 0.03
            messages.append("No action taken. Slight trust decay from inattention.")

        # ---- Tenant-type-specific modifiers ----
        if self._tenant_type == "loyal":
            # Loyal tenants are more tolerant; dampen trust changes
            trust_delta *= 0.5

        elif self._tenant_type == "price_sensitive":
            # Price-sensitive tenants punish rent increases harder
            if action.increase_rent:
                trust_delta -= 0.05  # Additional penalty

        elif self._tenant_type == "demanding":
            # Demanding tenants expect maintenance every 3 months
            if self._is_maintenance_due(month) and not action.perform_maintenance:
                trust_delta -= 0.30
                messages.append("WARNING: Maintenance ignored! Major trust drop.")

        # ---- Update trust (clamped to [0, 1]) ----
        self._trust_score = max(0.0, min(1.0, self._trust_score + trust_delta))

        # ---- Check vacancy conditions ----
        if self._tenant_type == "price_sensitive" and self._rent > self._market_rate * 1.10:
            self._is_vacant = True
            messages.append("TENANT LEFT: Rent exceeded market tolerance!")

        if self._trust_score < TRUST_FLOOR:
            self._is_vacant = True
            messages.append("TENANT LEFT: Trust dropped below threshold!")

        # ---- Calculate reward ----
        step_reward = self._rent - cost
        if self._is_vacant:
            step_reward -= VACANCY_PENALTY

        self._total_profit += self._rent
        self._total_cost += cost
        self._step_rewards.append(step_reward)

        # Update state totals
        self._state.total_profit = self._total_profit
        self._state.total_cost = self._total_cost

        done = self._is_vacant or self._state.step_count >= MAX_STEPS

        # Preview next month's maintenance status
        next_maintenance_due = (
            False if done else self._is_maintenance_due(month + 1)
        )

        return TenantObservation(
            done=done,
            reward=step_reward,
            rent=self._rent,
            trust_score=self._trust_score,
            tenant_type=self._tenant_type,
            months_stayed=self._state.step_count,
            is_vacant=self._is_vacant,
            market_rate=self._market_rate,
            maintenance_due=next_maintenance_due,
            message=" | ".join(messages) if messages else "Month passed uneventfully.",
        )

    # ------------------------------------------------------------------
    # OpenEnv interface: state (property)
    # ------------------------------------------------------------------

    @property
    def state(self) -> TenantState:
        """Return current episode metadata."""
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_maintenance_due(self, month: int) -> bool:
        """Demanding tenants expect maintenance every 3 months."""
        if self._tenant_type == "demanding":
            return month % 3 == 0
        return False

    def _terminal_observation(self, message: str) -> TenantObservation:
        """Return a done observation with zero reward."""
        return TenantObservation(
            done=True,
            reward=0.0,
            rent=self._rent,
            trust_score=self._trust_score,
            tenant_type=self._tenant_type,
            months_stayed=self._state.step_count,
            is_vacant=self._is_vacant,
            market_rate=self._market_rate,
            maintenance_due=False,
            message=message,
        )

    def get_step_rewards(self) -> list[float]:
        """Expose step rewards for grading."""
        return list(self._step_rewards)
