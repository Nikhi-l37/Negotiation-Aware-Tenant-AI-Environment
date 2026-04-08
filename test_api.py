import requests, json
r = requests.post('http://localhost:8000/reset')
print('RESET:', r.status_code)
r2 = requests.post('http://localhost:8000/step', json={'action': {'negotiate': True}})
print('STEP:', r2.status_code)
r3 = requests.get('http://localhost:8000/state')
print('STATE:', r3.status_code)
print('ALL ENDPOINTS OK' if all(x.status_code == 200 for x in [r, r2, r3]) else 'ENDPOINT FAILURE')
