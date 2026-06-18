from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    upstream_api_url: str = "https://alisaajer-newrepo18.hf.space/v1"
    api_key: str = ""
    tavily_api_key: str = ""

    # Automatically loads from a .env file if it exists
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _check_required(self) -> "Settings":
        if not self.upstream_api_url:
            raise ValueError(
                "UPSTREAM_API_URL is not set. "
                "Add it to your .env file or environment."
            )
        return self


settings = Settings()
