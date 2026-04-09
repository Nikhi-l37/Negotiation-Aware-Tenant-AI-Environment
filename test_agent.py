from server.environment import TenantNegotiationEnvironment
from models import TenantAction

env = TenantNegotiationEnvironment()

print('--- STARTING EPISODE (HARD TASK) ---')
obs = env.reset(task_name='hard')
print(f'Month 0: Rent ${obs.rent:.0f}, Market ${obs.market_rate:.0f}, Trust {obs.trust_score:.2f}')

print('\n--- MONTH 1: Action = negotiate ---')
obs = env.step(TenantAction(negotiate=True))
print(f'Rent: ${obs.rent:.0f}, Trust: {obs.trust_score:.2f}, Message: {obs.message}')

print('\n--- MONTH 2: Action = negotiate ---')
obs = env.step(TenantAction(negotiate=True))
print(f'Rent: ${obs.rent:.0f}, Trust: {obs.trust_score:.2f}, Message: {obs.message}')
print(f'[!] Next month maintenance due: {obs.maintenance_due}')

print('\n--- MONTH 3: Action = perform_maintenance (PIPE BURST & ROUTINE) ---')
obs = env.step(TenantAction(perform_maintenance=True))
print(f'Rent: ${obs.rent:.0f}, Trust: {obs.trust_score:.2f}, Message: {obs.message}')

print('\n--- SKIPPING TO MONTH 7 (MARKET BOOM) ---')
for _ in range(3): obs = env.step(TenantAction(negotiate=True, perform_maintenance=env._is_maintenance_due(env.state.step_count + 1)))
obs = env.step(TenantAction(negotiate=True))
print(f'Month {env.state.step_count}: Rent ${obs.rent:.0f}, Market ${obs.market_rate:.0f}')
print(f'Message: {obs.message}')

print('\n--- SKIPPING TO MONTH 10 (JOB LOSS) ---')
for _ in range(2): obs = env.step(TenantAction(negotiate=True, perform_maintenance=env._is_maintenance_due(env.state.step_count + 1)))
obs = env.step(TenantAction(increase_rent=False, negotiate=False, perform_maintenance=env._is_maintenance_due(env.state.step_count + 1)))
print(f'Month {env.state.step_count} (No rent increase): Trust: {obs.trust_score:.2f}')
print(f'Message: {obs.message}')

print('\n--- MONTH 11: MISTAKE (increase_rent) ---')
obs = env.step(TenantAction(increase_rent=True))
print(f'Trust: {obs.trust_score:.2f}, Vacant: {obs.is_vacant}')
print(f'Message: {obs.message}')
