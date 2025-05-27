from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # MongoDB settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "edith")
    
    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    
    # Anthropic settings
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # Google settings
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

    # Cerebras settings
    CEREBRAS_API_KEY: Optional[str] = os.getenv("CEREBRAS_API_KEY")

    # Openrouter settings
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    
    # Hugging Face settings
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_DEFAULT_MODEL: str = os.getenv("OLLAMA_DEFAULT_MODEL", "llama2")
    
    AWS_CDN_URL: Optional[str] = os.getenv("AWS_CDN_URL")
    AWS_BUCKET_NAME: Optional[str] = os.getenv("AWS_BUCKET_NAME")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_ENDPOINT_URL: Optional[str] = os.getenv("AWS_ENDPOINT_URL")
    
    # Application settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 