from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse, JSONResponse
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
                request.learningPrompt,
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
        response = await chat_service.generate_text_response(
            request.prompt,
            request.files,
            request.chatHistory,
            request.model,
            request.email,
            request.sessionId,
            request.reGenerate,
            request.chatType
        )
        return JSONResponse(
            content={
                "status": 200,
                "message": "Success",
                "data": response
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
    
@router.post("/chat/generateAudio")
async def chat_generate_audio(request: ChatRequest):
    try:
        response = await chat_service.generate_audio_response(
            request.prompt,
            request.files,
            request.chatHistory,
            request.model,
            request.email,
            request.sessionId,
            request.reGenerate,
            request.chatType,
        )
        return JSONResponse(
            content={
                "status": 200,
                "message": "Success",
                "data": response
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

@router.post("/chat/generateImage")
async def chat_generate_image(request: ChatRequest):
    try:
        response = await chat_service.generate_image_response(
            request.prompt,
            request.files,
            request.chatHistory,
            request.model,
            request.email,
            request.sessionId,
            request.reGenerate,
            request.chatType,
        )
        return JSONResponse(
            content={
                "status": 200,
                "message": "Success",
                "data": response
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

@router.get("/chat/image-support-info")
async def get_image_support_info():
    """
    Get information about supported image formats and usage guidelines.
    """
    try:
        info = chat_service.get_supported_image_info()
        return JSONResponse(
            content={
                "status": 200,
                "message": "Success",
                "data": info
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

