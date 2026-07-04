from pydantic_settings import BaseSettings
from functools import lru_cache


class settings(BaseSettings):
    """Application settings."""

    openai_api_key: str
    primary_model: str = "gpt-4o-mini"
    fallback_model: str = "gpt-3.5-turbo"

    langchain_tracking_v2: bool = True


    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: int = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3


    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def is_production(self) -> bool:
        """Check if the application is running in production."""
        return self.app_env == "production"


@lru_cache
def get_settings() -> settings:
    """Get the application settings."""
    return settings()