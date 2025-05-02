from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config.settings import settings
from .api.chat import router as chat_router

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