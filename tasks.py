"""
Task definitions and deterministic graders for the Tenant Negotiation Environment.

Each grader takes the environment instance after a full episode and returns
a float strictly in [0.0, 1.0].  Scores reflect partial progress — not binary.
"""

from server.environment import TenantNegotiationEnvironment, MAX_STEPS


# =====================================================================
# Easy Task — Loyal Tenant
# =====================================================================

def grade_easy(env: TenantNegotiationEnvironment) -> float:
    """
    Easy: Maximize profit with a loyal tenant.

    The loyal tenant rarely leaves (trust changes are dampened by 0.5×).
    A perfect agent maximises rent while keeping the tenant for all 12 months.

    Scoring breakdown:
        - 0.0  → tenant left (vacancy)
        - 0.0–0.3  → collected less than baseline
        - 0.3–0.7  → modest rent optimisation
        - 0.7–1.0  → strong rent optimisation with full retention
    """
    if env._is_vacant:
        return 0.0

    rewards = env.get_step_rewards()
    if not rewards:
        return 0.0

    # Baseline: $1500/mo × 12 = $18,000 (do-nothing)
    baseline = 1500.0 * MAX_STEPS
    # Theoretical max: aggressive negotiate every month ≈ $21,500
    ceiling = 22000.0

    total = sum(rewards)

    if total <= 0:
        return 0.0

    # Fraction of improvement over baseline, capped at ceiling
    score = (total - baseline) / (ceiling - baseline)
    return max(0.0, min(1.0, score))


# =====================================================================
# Medium Task — Price-Sensitive Tenant
# =====================================================================

def grade_medium(env: TenantNegotiationEnvironment) -> float:
    """
    Medium: Handle a price-sensitive tenant.

    The tenant will leave immediately if rent exceeds 110% of the $1,300 market
    rate ($1,430).  Starting rent is $1,200, starting trust is 0.8.
    The agent must raise rent carefully while sometimes performing maintenance
    or offering discounts to keep trust healthy.

    Scoring = 0.6 × profit_score + 0.4 × trust_score
    """
    if env._is_vacant:
        # Partial credit based on how many months survived
        months_survived = env.state.step_count
        survival_ratio = months_survived / MAX_STEPS
        return max(0.0, min(1.0, survival_ratio * 0.3))  # Max 0.3 for vacancy

    rewards = env.get_step_rewards()
    if not rewards:
        return 0.0

    # Baseline: $1200 × 12 = $14,400
    baseline = 1200.0 * MAX_STEPS
    # Ceiling: careful negotiate ≈ $16,500
    ceiling = 16800.0

    total = sum(rewards)
    profit_score = (total - baseline) / (ceiling - baseline) if ceiling > baseline else 0.0
    profit_score = max(0.0, min(1.0, profit_score))

    trust_score = max(0.0, min(1.0, env._trust_score))

    # Weighted combination
    score = 0.6 * profit_score + 0.4 * trust_score
    return max(0.0, min(1.0, score))


# =====================================================================
# Hard Task — Demanding Tenant
# =====================================================================

def grade_hard(env: TenantNegotiationEnvironment) -> float:
    """
    Hard: Handle a demanding tenant who requires maintenance every 3 months.

    Starting rent: $2,000, market: $2,200, initial trust: 0.7.
    Missing maintenance at months 3, 6, 9, 12 causes −0.30 trust each time,
    quickly leading to vacancy.  The agent must balance rent increases with
    mandatory maintenance while maintaining trust.

    Scoring = 0.4 × profit_score + 0.4 × trust_score + 0.2 × completion_bonus
    """
    months_survived = env.state.step_count
    completion_ratio = months_survived / MAX_STEPS

    if env._is_vacant:
        # Very harsh for vacancy on the hard task
        return max(0.0, min(1.0, completion_ratio * 0.2))

    rewards = env.get_step_rewards()
    if not rewards:
        return 0.0

    # Baseline: $2000 × 12 − 4 × $100 = $23,600  (4 mandatory maintenance months)
    baseline = 2000.0 * MAX_STEPS - 400.0
    # Ceiling: negotiate + maintain ≈ $25,800
    ceiling = 26000.0

    total = sum(rewards)
    profit_score = (total - baseline) / (ceiling - baseline) if ceiling > baseline else 0.0
    profit_score = max(0.0, min(1.0, profit_score))

    trust_score = max(0.0, min(1.0, env._trust_score))

    # Completion bonus rewards surviving the full episode
    completion_bonus = 1.0 if months_survived == MAX_STEPS else completion_ratio

    score = 0.4 * profit_score + 0.4 * trust_score + 0.2 * completion_bonus
    return max(0.0, min(1.0, score))
