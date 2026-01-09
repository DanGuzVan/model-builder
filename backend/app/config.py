from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # Application
    app_name: str = "NN Optimizer"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://nnoptimizer:nnoptimizer123@localhost:5432/nnoptimizer"
    database_url_sync: str = "postgresql://nnoptimizer:nnoptimizer123@localhost:5432/nnoptimizer"

    # Ollama LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"

    # File uploads
    upload_dir: str = "/app/uploads"
    max_upload_size: int = 50 * 1024 * 1024  # 50MB

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
