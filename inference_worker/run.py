import uvicorn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

if __name__ == "__main__":
    uvicorn.run(
        "inference_worker.main:app",
        host=settings.worker_host,
        port=settings.worker_port,
        reload=False,       # reload=False — model loading is expensive
        log_level="info",
        timeout_keep_alive=120,
    )
