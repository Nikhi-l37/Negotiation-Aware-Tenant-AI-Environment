"""
Typed HTTP/WebSocket client for the Negotiation-Aware Tenant AI Environment.

This is what users import to interact with a remote (or local) deployment
of the environment. Follows the OpenEnv 3-abstract-method pattern.
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import TenantAction, TenantObservation, TenantState


class TenantEnv(EnvClient[TenantAction, TenantObservation, TenantState]):
    """
    Client for the Negotiation-Aware Tenant AI Environment.

    Usage (async):
        async with TenantEnv(base_url="https://your-space.hf.space") as env:
            result = await env.reset(task_name="easy")
            result = await env.step(TenantAction(negotiate=True))

    Usage (sync):
        with TenantEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(task_name="easy")
            result = env.step(TenantAction(negotiate=True))
    """

    def _step_payload(self, action: TenantAction) -> dict:
        """Convert a TenantAction to the JSON payload for the server."""
        return {
            "increase_rent": action.increase_rent,
            "offer_discount": action.offer_discount,
            "perform_maintenance": action.perform_maintenance,
            "negotiate": action.negotiate,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse the server JSON response into a StepResult."""
        obs_data = payload.get("observation", payload)
        return StepResult(
            observation=TenantObservation(
                done=payload.get("done", obs_data.get("done", False)),
                reward=payload.get("reward", obs_data.get("reward")),
                rent=obs_data.get("rent", 0.0),
                trust_score=obs_data.get("trust_score", 1.0),
                tenant_type=obs_data.get("tenant_type", "loyal"),
                months_stayed=obs_data.get("months_stayed", 0),
                is_vacant=obs_data.get("is_vacant", False),
                market_rate=obs_data.get("market_rate", 0.0),
                maintenance_due=obs_data.get("maintenance_due", False),
                message=obs_data.get("message", ""),
            ),
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> TenantState:
        """Parse the server JSON response into a TenantState."""
        return TenantState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", "easy"),
            total_profit=payload.get("total_profit", 0.0),
            total_cost=payload.get("total_cost", 0.0),
            max_steps=payload.get("max_steps", 12),
        )
