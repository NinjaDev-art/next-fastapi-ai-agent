from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class IRouterChatLog(BaseModel):
    prompt: str
    response: Optional[str] = None

class ChatRequest(BaseModel):
    prompt: str
    sessionId: str = ""
    chatHistory: List[IRouterChatLog] = []
    files: List[str] = []
    email: str = ""
    reGenerate: bool = False
    model: str = "gpt-3.5-turbo"
    chatType: int = 0
    points: float = 0

class AiConfig(BaseModel):
    name: str
    inputCost: float
    outputCost: float
    multiplier: float
    model: str
    provider: str 