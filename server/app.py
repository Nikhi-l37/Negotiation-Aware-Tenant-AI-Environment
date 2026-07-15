"""
FastAPI server for the Negotiation-Aware Tenant AI Environment.

One-liner that wraps the environment class in the standard OpenEnv HTTP/WS server.
Run with: uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server import create_app
from .environment import TenantNegotiationEnvironment
from models import TenantAction, TenantObservation
import uvicorn

app = create_app(TenantNegotiationEnvironment, TenantAction, TenantObservation)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
