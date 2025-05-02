from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class IRouterChatLog(BaseModel):
    prompt: str
    response: Optional[str] = None

class ChatRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"
    reGenerate: bool = False
    files: List[str] = []
    chatHistory: List[IRouterChatLog] = []
    sessionId: str = ""
    email: str = ""

class AiConfig(BaseModel):
    name: str
    inputCost: float
    outputCost: float
    multiplier: float
    model: str
    provider: str 