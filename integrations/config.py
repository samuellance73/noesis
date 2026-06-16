from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    upstream_api_url: str = "https://alisaajer-newrepo18.hf.space/v1"
    api_key: str = ""

    # Automatically loads from a .env file if it exists
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
