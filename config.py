from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    router_host: str = "0.0.0.0"
    router_port: int = 8001

    worker_host: str = "0.0.0.0"
    worker_port: int = 8002

    default_model: str = "gpt2"
    large_model: str = "gpt2-medium"
    small_model: str = "gpt2"

    prometheus_port: int = 9090
    metrics_port: int = 8003

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
