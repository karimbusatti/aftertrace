from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    
    app_name: str = "Aftertrace"
    debug: bool = False
    # CORS_ORIGINS env var: comma-separated list of allowed origins
    # e.g. "https://aftertrace.vercel.app,http://localhost:3000"
    cors_origins_raw: str = Field(default="http://localhost:3000", validation_alias="CORS_ORIGINS")
    
    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        """Parse comma-separated CORS origins into list."""
        origins = [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]
        # Fall back to localhost if env var is empty/whitespace-only
        return origins if origins else ["http://localhost:3000"]


settings = Settings()

