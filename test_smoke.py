"""Quick smoke test to verify environment mechanics work end-to-end."""
import sys
sys.path.insert(0, ".")

from server.environment import TenantNegotiationEnvironment
from models import TenantAction
from tasks import grade_easy, grade_medium, grade_hard

def test_task(task_name, action_fn, grader):
    env = TenantNegotiationEnvironment()
    obs = env.reset(task_name=task_name)
    print(f"[{task_name.upper()}] Reset: rent={obs.rent}, trust={obs.trust_score}, type={obs.tenant_type}")
    for i in range(12):
        action = action_fn(i)
        obs = env.step(action)
        print(f"  Step {i+1}: rent=${obs.rent:.0f}, trust={obs.trust_score:.2f}, reward={obs.reward:.0f}, done={obs.done}")
        if obs.done:
            break
    score = grader(env)
    print(f"[{task_name.upper()}] Final score: {score:.4f}")
    print()

# Easy: just negotiate every month
test_task("easy", lambda i: TenantAction(negotiate=True), grade_easy)

# Medium: negotiate + occasional maintenance
test_task("medium", lambda i: TenantAction(negotiate=True, perform_maintenance=(i%4==0)), grade_medium)

# Hard: negotiate + mandatory maintenance on months 3,6,9,12
test_task("hard", lambda i: TenantAction(negotiate=True, perform_maintenance=((i+1)%3==0)), grade_hard)

print("ALL TESTS PASSED!")
