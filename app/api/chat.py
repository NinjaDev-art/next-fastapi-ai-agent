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
                request.reGenerate
            ),
            media_type="text/event-stream"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 