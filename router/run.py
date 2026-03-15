import uvicorn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

if __name__ == "__main__":
    uvicorn.run(
        "router.main:app",
        host=settings.router_host,
        port=settings.router_port,
        reload=True,
        log_level="info"
    )
