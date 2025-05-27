from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config.settings import settings
from .api.chat import router as chat_router
from .config.logging_config import setup_logging
import logging

# Setup logging first
setup_logging()
logger = logging.getLogger(__name__)

# Verify environment variables
logger.info(f"MONGODB_URI: {settings.MONGODB_URI}")
logger.info(f"DB_NAME: {settings.DB_NAME}")
logger.info(f"OPENAI_API_KEY: {'Set' if settings.OPENAI_API_KEY else 'Not Set'}")
logger.info(f"ANTHROPIC_API_KEY: {'Set' if settings.ANTHROPIC_API_KEY else 'Not Set'}")
logger.info(f"DEFAULT_MODEL: {settings.DEFAULT_MODEL}")
logger.info(f"EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
logger.info(f"AWS_CDN_URL: {settings.AWS_CDN_URL}")
logger.info(f"DEBUG: {settings.DEBUG}")
logger.info(f"CORS_ORIGINS: {settings.CORS_ORIGINS}")

app = FastAPI(
    title="Chatbot API",
    description="A FastAPI-based chatbot with RAG capabilities",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API"} 