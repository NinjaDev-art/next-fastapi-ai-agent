from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # MongoDB settings
    MONGODB_URI: str = "mongodb://localhost:27017"
    DB_NAME: str = "edith"
    
    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = None
    DEFAULT_MODEL: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    AWS_CDN_URL: Optional[str] = None
    
    # Application settings
    DEBUG: bool = False
    CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 