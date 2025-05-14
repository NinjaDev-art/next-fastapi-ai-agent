from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse
from ..models.chat import ChatRequest
from ..services.chat_service import chat_service

router = APIRouter()

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    try:
        return StreamingResponse(
            chat_service.generate_stream_response(
                request.prompt,
                request.files,
                request.chatHistory,
                request.model,
                request.email,
                request.sessionId,
                request.reGenerate,
                request.chatType,
                request.points
            ),
            media_type="text/event-stream"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
    
@router.post("/chat/generateText")
async def chat_generate_text(request: ChatRequest):
    try:
        return chat_service.generate_text_response(
            request.prompt,
            request.files,
            request.chatHistory,
            request.model,
            request.email,
            request.sessionId,
            request.reGenerate,
            request.chatType,
            request.points
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 