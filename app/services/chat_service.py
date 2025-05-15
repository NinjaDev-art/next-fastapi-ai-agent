from typing import AsyncGenerator, List, Optional
import asyncio
from datetime import datetime
import logging
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from langchain_cerebras import ChatCerebras
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from fastapi import HTTPException

from ..config.settings import settings
from ..core.database import db
from ..models.chat import IRouterChatLog, AiConfig
from ..utils.file_processor import file_processor
from ..utils.user_point import user_point

from openai import OpenAI
import boto3
import os
from botocore.config import Config
import tempfile

logger = logging.getLogger(__name__)

class NoPointsAvailableException(Exception):
    pass

class ChatService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
        )
        self.vector_stores = {}
        self.encoding = None  # Will be set based on the model being used
        self.openai = OpenAI(api_key=settings.OPENAI_API_KEY)
        
    def get_chat_messages(self, chat_history: List[IRouterChatLog], provider: str = "openai") -> List[dict]:
        system_prompt = "" if provider == "edith" else db.get_system_prompt()
        messages = []
        
        # For Anthropic, we don't include system message in the messages array
        if provider.lower() != "anthropic":
            messages.append({"role": "system", "content": system_prompt})
            
        for chat in chat_history:
            if chat.prompt:
                messages.append({"role": "user", "content": chat.prompt})
            if chat.response:
                messages.append({"role": "assistant", "content": chat.response})
                
        return messages, system_prompt

    def get_points(self, inputToken: int, outputToken: int, ai_config: AiConfig) -> float:
        print("inputToken", inputToken)
        print("outputToken", outputToken)
        print("ai_config", ai_config)
        return (inputToken * ai_config.inputCost + outputToken * ai_config.outputCost) * ai_config.multiplier / 0.001

    def _get_llm(self, ai_config: AiConfig, isStream: bool = True):
        if ai_config.provider.lower() == "anthropic":
            return ChatAnthropic(
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                stream_options={"include_usage": isStream}
            )
        elif ai_config.provider.lower() == "deepseek":
            return ChatDeepSeek(
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                deepseek_api_key=settings.DEEPSEEK_API_KEY,
                stream_options={"include_usage": isStream}
            )
        elif ai_config.provider.lower() == "google":
            return ChatGoogleGenerativeAI(
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                google_api_key=settings.GOOGLE_API_KEY,
                stream_options={"include_usage": isStream}
            )
        elif ai_config.provider.lower() == "xai":
            return ChatXAI(   
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                xai_api_key=settings.XAI_API_KEY,
                stream_usage=isStream
            )
        elif ai_config.provider.lower() == "ollama":
            return ChatOllama(
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                base_url=settings.OLLAMA_BASE_URL,
                stream_usage=isStream
            )
        elif ai_config.provider.lower() == "mistralai":
            return ChatMistralAI(
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                mistral_api_key=settings.MISTRAL_API_KEY,
                stream_usage=isStream
            )
        elif ai_config.provider.lower() == "cerebras" or ai_config.provider.lower() == "edith":
            return ChatCerebras(
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model="llama3.1-8b" if ai_config.provider.lower() == "edith" else ai_config.model,
                cerebras_api_key=settings.CEREBRAS_API_KEY,
                stream_usage=isStream
            )
        else:  # Default to OpenAI
            return ChatOpenAI(
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                openai_api_key=settings.OPENAI_API_KEY,
                stream_usage=isStream
            )

    async def generate_stream_response(
        self,
        query: str,
        files: List[str],
        chat_history: List[IRouterChatLog],
        model: str,
        email: str,
        sessionId: str,
        reGenerate: bool,
        chatType: int,
    ) -> AsyncGenerator[str, None]:
        logger.info(f"Generating response for query: {query}")
        full_response = ""
        points = 0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        outputTime = datetime.now()

        try:
            # Initialize user_point with email
            await user_point.initialize(email)
            
            ai_config = db.get_ai_config(model)
            print("ai_config", ai_config)
            if not ai_config:
                yield "Error: Invalid AI configuration"
                return

            if files:
                print("Using RAG with files")
                vector_store = self._get_vector_store(files)
                
                # Get messages for token estimation
                messages, system_prompt = self.get_chat_messages(chat_history, ai_config.provider)
                messages.append({"role": "user", "content": query})
                
                # system_template = (system_prompt or db.get_system_prompt()) + "\n\nPrevious conversation:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}"
                system_template = system_prompt + "\n\nPrevious conversation:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}"
                
                # Estimate tokens before making the API call
                estimated_tokens = self.estimate_total_tokens(messages, system_template, "llama3.1-8b" if ai_config.provider.lower() == "edith" else ai_config.model)
                estimated_points = self.get_points(estimated_tokens["prompt_tokens"], estimated_tokens["completion_tokens"], ai_config)
                print(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")
                
                check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points)
                if not check_user_available_to_chat:
                    error_response = {
                        "error": True,
                        "status": 429,
                        "message": "Insufficient points available",
                        "details": {
                            "estimated_points": estimated_points,
                            "available_points": user_point.user_doc.get("availablePoints", 0) if user_point.user_doc else 0,
                            "points_used": user_point.user_doc.get("pointsUsed", 0) if user_point.user_doc else 0
                        }
                    }
                    yield f"\n\n[ERROR]{error_response}"
                    return
                
                llm = self._get_llm(ai_config)
                
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(system_template),
                    HumanMessagePromptTemplate.from_template("{question}")
                ])
                
                rag_chain = (
                    {
                        "context": vector_store.as_retriever(),
                        "question": RunnablePassthrough(),
                        "chat_history": lambda x: "\n".join([f"User: {h.prompt}\nAssistant: {h.response}" for h in chat_history if h.response])
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                async for event in rag_chain.astream_events(query):
                    if event["event"] == "on_chat_model_end":
                        usage = event["data"]["output"].usage_metadata
                        if usage:
                            token_usage = {
                                "prompt_tokens": usage.get("input_tokens", 0),
                                "completion_tokens": usage.get("output_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0)
                            }
                            points = self.get_points(token_usage["prompt_tokens"], token_usage["completion_tokens"], ai_config)
                            logger.info(f"Token usage for RAG: {token_usage}, Cost: ${points:.6f}")
                    elif event["event"] == "on_chain_stream" and event["name"] == "RunnableSequence":
                        try:
                            chunk = event["data"]["chunk"]
                            if chunk is None:
                                continue
                            chunk_str = str(chunk) if not isinstance(chunk, str) else chunk
                            full_response += chunk_str
                            yield chunk_str
                        except Exception as e:
                            logger.error(f"Error processing chunk: {str(e)}")
                            continue
                        await asyncio.sleep(0.01)
                
                points = self.get_points(token_usage["prompt_tokens"], token_usage["completion_tokens"], ai_config)
                yield f"\n\n[POINTS]{points}"
                self.remove_vector_store()
                
            else:
                print(f"Using direct {ai_config.provider} completion")
                llm = self._get_llm(ai_config)
                messages, system_prompt = self.get_chat_messages(chat_history, ai_config.provider)
                messages.append({"role": "user", "content": query})
                
                # Estimate tokens before making the API call
                #system_template = system_prompt or db.get_system_prompt()
                system_template = system_prompt
                estimated_tokens = self.estimate_total_tokens(messages, system_template, "llama3.1-8b" if ai_config.provider.lower() == "edith" else ai_config.model)
                estimated_points = self.get_points(estimated_tokens["prompt_tokens"], estimated_tokens["completion_tokens"], ai_config)
                print(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")
                
                check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points)
                if not check_user_available_to_chat:
                    error_response = {
                        "error": True,
                        "status": 429,
                        "message": "Insufficient points available",
                        "details": {
                            "estimated_points": estimated_points,
                            "available_points": user_point.user_doc.get("availablePoints", 0) if user_point.user_doc else 0,
                            "points_used": user_point.user_doc.get("pointsUsed", 0) if user_point.user_doc else 0
                        }
                    }
                    yield f"\n\n[ERROR]{error_response}"
                    return
                
                
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(system_template),
                    HumanMessagePromptTemplate.from_template("{question}")
                ])
                
                chain = (
                    {"question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                async for event in chain.astream_events(query):
                    if event["event"] == "on_chat_model_end":
                        usage = event["data"]["output"].usage_metadata
                        if usage:
                            token_usage = {
                                "prompt_tokens": usage.get("input_tokens", 0),
                                "completion_tokens": usage.get("output_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0)
                            }
                            points = self.get_points(token_usage["prompt_tokens"], token_usage["completion_tokens"], ai_config)
                            logger.info(f"Token usage for completion: {token_usage}, Cost: ${points:.6f}")
                    elif event["event"] == "on_chain_stream" and event["name"] == "RunnableSequence":
                        try:
                            chunk = event["data"]["chunk"]
                            if chunk is None:
                                continue
                            chunk_str = str(chunk) if not isinstance(chunk, str) else chunk
                            full_response += chunk_str
                            yield chunk_str
                        except Exception as e:
                            logger.error(f"Error processing chunk: {str(e)}")
                            continue
                        await asyncio.sleep(0.01)
                
                points = self.get_points(token_usage["prompt_tokens"], token_usage["completion_tokens"], ai_config)
                yield f"\n\n[POINTS]{points}"
            
            outputTime = (datetime.now() - outputTime).total_seconds()
            yield f"\n\n[OUTPUT_TIME]{outputTime}"
            
            await db.save_chat_log({
                "email": email,
                "sessionId": sessionId,
                "reGenerate": reGenerate,
                "title": full_response.split("\n\n")[0],
                "chat": {
                    "prompt": query,
                    "response": full_response,
                    "timestamp": datetime.now(),
                    "inputToken": token_usage["prompt_tokens"],
                    "outputToken": token_usage["completion_tokens"],
                    "outputTime": outputTime,
                    "chatType": chatType,
                    "fileUrls": files,
                    "model": model,
                    "points": points
                }
            })
            await db.save_usage_log({
                "date": datetime.now(),
                "userId": user_point.user_doc.get("_id", None),
                "modelId": model,
                "planId": user_point.user_doc.get("currentplan", "680f11c0d44970f933ae5e54"),
                "stats": {
                    "tokenUsage": {
                        "input": token_usage["prompt_tokens"],
                        "output": token_usage["completion_tokens"],
                        "total": token_usage["total_tokens"]
                    },
                    "pointsUsage": points
                }
            })
            await user_point.save_user_points(points)
        except Exception as e:
            logger.error(f"Error in generate_stream_response: {str(e)}")
            error_response = {
                "error": True,
                "status": 500,
                "message": "An error occurred while processing your request",
                "details": str(e)
            }
            yield f"\n\n[ERROR]{error_response}"

    async def generate_text_response(
        self,
        query: str,
        files: List[str],
        chat_history: List[IRouterChatLog],
        model: str,
        email: str,
        sessionId: str,
        reGenerate: bool,
        chatType: int,
    ) -> str:
        logger.info(f"Generating response for query: {query}")
        points = 0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        outputTime = datetime.now()
        full_response = ""

        try:
            # Initialize user_point with email
            await user_point.initialize(email)
            ai_config = db.get_ai_config(model)

            print("ai_config", ai_config)
            if not ai_config:
                return "Error: Invalid AI configuration"

            if files:
                print("Using RAG with files")
                llm = self._get_llm(ai_config, False)
                vector_store = self._get_vector_store(files)
                
                # Get messages for token estimation
                messages, system_prompt = self.get_chat_messages(chat_history, ai_config.provider)
                messages.append({"role": "user", "content": query})
                
                system_template = system_prompt + "\n\nPrevious conversation:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}"
                
                # Estimate tokens before making the API call
                estimated_tokens = self.estimate_total_tokens(messages, system_template, "llama3.1-8b" if ai_config.provider.lower() == "edith" else ai_config.model)
                estimated_points = self.get_points(estimated_tokens["prompt_tokens"], estimated_tokens["completion_tokens"], ai_config)
                print(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")
                
                check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points)
                if not check_user_available_to_chat:
                    error_response = {
                        "error": True,
                        "status": 429,
                        "message": "Insufficient points available",
                        "details": {
                            "estimated_points": estimated_points,
                            "available_points": user_point.user_doc.get("availablePoints", 0) if user_point.user_doc else 0,
                            "points_used": user_point.user_doc.get("pointsUsed", 0) if user_point.user_doc else 0
                        }
                    }
                    return f"\n\n[ERROR]{error_response}"
                
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(system_template),
                    HumanMessagePromptTemplate.from_template("{question}")
                ])
                
                rag_chain = (
                    {
                        "context": vector_store.as_retriever(),
                        "question": RunnablePassthrough(),
                        "chat_history": lambda x: "\n".join([f"User: {h.prompt}\nAssistant: {h.response}" for h in chat_history if h.response])
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                ai_response = rag_chain.invoke(query)
                full_response = ai_response.content
                
                # Track actual token usage
                token_usage = self.track_actual_token_usage(messages, full_response, "llama3.1-8b" if ai_config.provider.lower() == "edith" else ai_config.model)
                
                outputTime = (datetime.now() - outputTime).total_seconds()
                points = self.get_points(token_usage["prompt_tokens"], token_usage["completion_tokens"], ai_config)
                response = f"{full_response}\n\n[POINTS]{points}\n\n[OUTPUT_TIME]{outputTime}"
                self.remove_vector_store()

            else:
                llm = self._get_llm(ai_config, False)
                print(f"Using direct {ai_config.provider} completion")
                messages, system_prompt = self.get_chat_messages(chat_history, ai_config.provider)
                messages.append({"role": "user", "content": query})
                
                system_template = system_prompt
                estimated_tokens = self.estimate_total_tokens(messages, system_template, "llama3.1-8b" if ai_config.provider.lower() == "edith" else ai_config.model)
                estimated_points = self.get_points(estimated_tokens["prompt_tokens"], estimated_tokens["completion_tokens"], ai_config)
                print(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")
                
                check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points)
                if not check_user_available_to_chat:
                    error_response = {
                        "error": True,
                        "status": 429,
                        "message": "Insufficient points available",
                        "details": {
                            "estimated_points": estimated_points,
                            "available_points": user_point.user_doc.get("availablePoints", 0) if user_point.user_doc else 0,
                            "points_used": user_point.user_doc.get("pointsUsed", 0) if user_point.user_doc else 0
                        }
                    }
                    return f"\n\n[ERROR]{error_response}"
                
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(system_template),
                    HumanMessagePromptTemplate.from_template("{question}")
                ])
                
                chain = (
                    {"question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                ai_response = chain.invoke(query)
                full_response = ai_response
                
                # Track actual token usage
                token_usage = self.track_actual_token_usage(messages, full_response, "llama3.1-8b" if ai_config.provider.lower() == "edith" else ai_config.model)
                
                outputTime = (datetime.now() - outputTime).total_seconds()
                points = self.get_points(token_usage["prompt_tokens"], token_usage["completion_tokens"], ai_config)
                response = f"{full_response}\n\n[POINTS]{points}\n\n[OUTPUT_TIME]{outputTime}"
            
            await db.save_chat_log({
                "email": email,
                "sessionId": sessionId,
                "reGenerate": reGenerate,
                "title": full_response.split("\n\n")[0] if full_response else "",
                "chat": {
                    "prompt": query,
                    "response": full_response,
                    "timestamp": datetime.now(),
                    "inputToken": token_usage["prompt_tokens"],
                    "outputToken": token_usage["completion_tokens"],
                    "outputTime": outputTime,
                    "chatType": chatType,
                    "fileUrls": files,
                    "model": model,
                    "points": points
                }
            })
            await db.save_usage_log({
                "date": datetime.now(),
                "userId": user_point.user_doc.get("_id", None),
                "modelId": model,
                "planId": user_point.user_doc.get("currentplan", "680f11c0d44970f933ae5e54"),
                "stats": {
                    "tokenUsage": {
                        "input": token_usage["prompt_tokens"],
                        "output": token_usage["completion_tokens"],
                        "total": token_usage["total_tokens"]
                    },
                    "pointsUsage": points
                }
            })
            await user_point.save_user_points(points)
            return response
        except Exception as e:
            logger.error(f"Error in generate_text_response: {str(e)}")
            error_response = {
                "error": True,
                "status": 500,
                "message": "An error occurred while processing your request",
                "details": str(e)
            }
            return f"\n\n[ERROR]{error_response}"

    def estimate_image_tokens(self, prompt: str) -> dict:
        """
        Estimate token usage for DALL-E 2 image generation.
        DALL-E 2 uses approximately 1 token per 4 characters for the prompt.
        """
        try:
            # Rough estimation: 1 token per 4 characters
            char_count = len(prompt)
            estimated_tokens = char_count // 4
            
            return {
                "prompt_tokens": estimated_tokens,
                "completion_tokens": 0,  # No completion tokens for image generation
                "total_tokens": estimated_tokens
            }
        except Exception as e:
            logger.error(f"Error estimating image tokens: {str(e)}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def generate_image_response(
        self,
        query: str,
        files: List[str],
        chat_history: List[IRouterChatLog],
        model: str,
        email: str,
        sessionId: str,
        reGenerate: bool,
        chatType: int,
    ) -> str:
        logger.info(f"Generating response for query: {query}")
        points = 0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        outputTime = datetime.now()
        full_response = ""

        try:
            # Initialize user_point with email
            await user_point.initialize(email)
            ai_config = db.get_ai_config(model)

            print("ai_config", ai_config)
            if not ai_config:
                return "Error: Invalid AI configuration"
            
            # Get messages for token estimation
            messages, system_prompt = self.get_chat_messages(chat_history, ai_config.provider)
            messages.append({"role": "user", "content": query})
            # Process files if they exist
            if files:
                print("Using RAG with files")
                vector_store = self._get_vector_store(files)
                # Get relevant context from files
                retriever = vector_store.as_retriever()
                docs = retriever.get_relevant_documents(query)
                context = "\n".join([doc.page_content for doc in docs])
                # Enhance the prompt with context
                enhanced_query = f"Context from files:\n{context}\n\nGenerate image based on this context and the following description: {query}"
            else:
                enhanced_query = query

            # Estimate tokens for the prompt
            estimated_tokens = self.estimate_image_tokens(enhanced_query)  # Use image-specific token estimation
            estimated_points = self.get_points(estimated_tokens["prompt_tokens"], estimated_tokens["completion_tokens"], ai_config)
            print(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")

            # Check if user has enough points
            check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points)
            if not check_user_available_to_chat:
                error_response = {
                    "error": True,
                    "status": 429,
                    "message": "Insufficient points available",
                    "details": {
                        "estimated_points": estimated_points,
                        "available_points": user_point.user_doc.get("availablePoints", 0) if user_point.user_doc else 0,
                        "points_used": user_point.user_doc.get("pointsUsed", 0) if user_point.user_doc else 0
                    }
                }
                return f"\n\n[ERROR]{error_response}"

            # Generate image using DALL-E 2
            response = self.openai.images.generate(
                model="dall-e-2",
                prompt=enhanced_query,
                n=1,
                size="1024x1024"
            )

            # Get the image URL
            image_url = response.data[0].url
            
            # Convert image URL to base64
            try:
                import requests
                import tempfile
                import os
                
                # Download the image
                image_response = requests.get(image_url)
                image_response.raise_for_status()
                
                # Create a temporary directory for image files
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save the image file temporarily
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_image_path = os.path.join(temp_dir, f"image_{timestamp}.png")
                    
                    # Write the response content to file
                    with open(temp_image_path, 'wb') as f:
                        f.write(image_response.content)

                    # Initialize S3 client for DigitalOcean Spaces
                    s3_client = boto3.client('s3',
                        endpoint_url=settings.AWS_ENDPOINT_URL,
                        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                        config=Config(s3={'addressing_style': 'virtual'})
                    )

                    # Upload to DigitalOcean Spaces
                    bucket_name = settings.AWS_BUCKET_NAME
                    cdn_url = settings.AWS_CDN_URL
                    object_key = f"images/image_{timestamp}.png"
                    
                    s3_client.upload_file(
                        temp_image_path,
                        bucket_name,
                        object_key,
                        ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/png'}
                    )

                    # Generate the CDN URL
                    full_response = f"{cdn_url}/{object_key}"
            except Exception as e:
                logger.error(f"Error converting image to base64: {str(e)}")
                full_response = image_url  # Fallback to original URL if conversion fails

            print(f"full_response: {response}")
            
            if response.usage != None:
                token_usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            else:
                token_usage = estimated_tokens

            points = self.get_points(token_usage["prompt_tokens"], token_usage["completion_tokens"], ai_config)
            outputTime = (datetime.now() - outputTime).total_seconds()
            response = f"{full_response}\n\n[POINTS]{points}\n\n[OUTPUT_TIME]{outputTime}"

            # Save chat log and usage
            await db.save_chat_log({
                "email": email,
                "sessionId": sessionId,
                "reGenerate": reGenerate,
                "title": "Image Generation",
                "chat": {
                    "prompt": query,
                    "response": full_response,
                    "timestamp": datetime.now(),
                    "inputToken": token_usage["prompt_tokens"],
                    "outputToken": token_usage["completion_tokens"],
                    "outputTime": outputTime,
                    "chatType": chatType,
                    "fileUrls": files,
                    "model": model,
                    "points": points
                }
            })

            await db.save_usage_log({
                "date": datetime.now(),
                "userId": user_point.user_doc.get("userId", None),
                "modelId": model,
                "planId": user_point.user_doc.get("currentplan", "680f11c0d44970f933ae5e54"),
                "stats": {
                    "tokenUsage": {
                        "input": token_usage["prompt_tokens"],
                        "output": token_usage["completion_tokens"],
                        "total": token_usage["total_tokens"]
                    },
                    "pointsUsage": points
                }
            })

            await user_point.save_user_points(points)
            return response

        except Exception as e:
            logger.error(f"Error in generate_image_response: {str(e)}")
            error_response = {
                "error": True,
                "status": 500,
                "message": "An error occurred while processing your request",
                "details": str(e)
            }
            return f"\n\n[ERROR]{error_response}"
        
    def estimate_audio_tokens(self, text: str) -> dict:
        """
        Estimate token usage for audio generation based on text length.
        GPT-4o-mini-tts uses approximately 1 token per 4 characters.
        """
        try:
            # Rough estimation: 1 token per 4 characters
            char_count = len(text)
            estimated_tokens = char_count // 4
            
            return {
                "prompt_tokens": estimated_tokens,
                "completion_tokens": 0,  # No completion tokens for audio generation
                "total_tokens": estimated_tokens
            }
        except Exception as e:
            logger.error(f"Error estimating audio tokens: {str(e)}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def generate_audio_response(
        self,
        query: str,
        files: List[str],
        chat_history: List[IRouterChatLog],
        model: str,
        email: str,
        sessionId: str,
        reGenerate: bool,
        chatType: int,
    ) -> str:
        logger.info(f"Generating response for query: {query}")
        points = 0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        outputTime = datetime.now()
        full_response = ""

        try:
            # Initialize user_point with email
            await user_point.initialize(email)
            ai_config = db.get_ai_config(model)

            print("ai_config", ai_config)
            if not ai_config:
                return "Error: Invalid AI configuration"

            # Get messages for token estimation
            messages, system_prompt = self.get_chat_messages(chat_history, ai_config.provider)
            messages.append({"role": "user", "content": query})

            # Process files if they exist
            if files:
                print("Using RAG with files")
                vector_store = self._get_vector_store(files)
                # Get relevant context from files
                retriever = vector_store.as_retriever()
                docs = retriever.get_relevant_documents(query)
                context = "\n".join([doc.page_content for doc in docs])
                # Enhance the prompt with context
                enhanced_query = f"Context from files:\n{context}\n\nGenerate audio based on this context and the following text: {query}"
            else:
                enhanced_query = query

            # Estimate tokens for audio generation
            estimated_tokens = self.estimate_audio_tokens(enhanced_query)
            estimated_points = self.get_points(estimated_tokens["prompt_tokens"], estimated_tokens["completion_tokens"], ai_config)
            print(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")

            # Check if user has enough points
            check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points)
            if not check_user_available_to_chat:
                error_response = {
                    "error": True,
                    "status": 429,
                    "message": "Insufficient points available",
                    "details": {
                        "estimated_points": estimated_points,
                        "available_points": user_point.user_doc.get("availablePoints", 0) if user_point.user_doc else 0,
                        "points_used": user_point.user_doc.get("pointsUsed", 0) if user_point.user_doc else 0
                    }
                }
                return f"\n\n[ERROR]{error_response}"

            # Generate audio using GPT-4o-mini-tts
            response = self.openai.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",  # You can also use "echo", "fable", "onyx", "nova", "shimmer"
                input=enhanced_query
            )

            # Create a temporary directory for audio files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save the audio file temporarily
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_audio_path = os.path.join(temp_dir, f"audio_{timestamp}.mp3")
                
                # Write the response content to file
                with open(temp_audio_path, 'wb') as f:
                    f.write(response.content)

                # Initialize S3 client for DigitalOcean Spaces
                s3_client = boto3.client('s3',
                    endpoint_url=settings.AWS_ENDPOINT_URL,
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    config=Config(s3={'addressing_style': 'virtual'})
                )

                # Upload to DigitalOcean Spaces
                bucket_name = settings.AWS_BUCKET_NAME
                cdn_url = settings.AWS_CDN_URL
                object_key = f"audio/audio_{timestamp}.mp3"
                
                s3_client.upload_file(
                    temp_audio_path,
                    bucket_name,
                    object_key,
                    ExtraArgs={'ACL': 'public-read', 'ContentType': 'audio/mpeg'}
                )

                # Generate the CDN URL
                audio_url = f"{cdn_url}/{object_key}"
                full_response = audio_url

            # Calculate actual token usage
            token_usage = self.estimate_audio_tokens(enhanced_query)
            
            # Calculate points for audio generation
            points = self.get_points(token_usage["prompt_tokens"], token_usage["completion_tokens"], ai_config)

            outputTime = (datetime.now() - outputTime).total_seconds()
            response = f"{full_response}\n\n[POINTS]{points}\n\n[OUTPUT_TIME]{outputTime}"

            # Save chat log and usage
            await db.save_chat_log({
                "email": email,
                "sessionId": sessionId,
                "reGenerate": reGenerate,
                "title": "Audio Generation",
                "chat": {
                    "prompt": query,
                    "response": full_response,
                    "timestamp": datetime.now(),
                    "inputToken": token_usage["prompt_tokens"],
                    "outputToken": token_usage["completion_tokens"],
                    "outputTime": outputTime,
                    "chatType": chatType,
                    "fileUrls": files,
                    "model": model,
                    "points": points
                }
            })

            await db.save_usage_log({
                "date": datetime.now(),
                "userId": user_point.user_doc.get("_id", None),
                "modelId": model,
                "planId": user_point.user_doc.get("currentplan", "680f11c0d44970f933ae5e54"),
                "stats": {
                    "tokenUsage": {
                        "input": token_usage["prompt_tokens"],
                        "output": token_usage["completion_tokens"],
                        "total": token_usage["total_tokens"]
                    },
                    "pointsUsage": points
                }
            })

            await user_point.save_user_points(points)
            return response

        except Exception as e:
            logger.error(f"Error in generate_audio_response: {str(e)}")
            error_response = {
                "error": True,
                "status": 500,
                "message": "An error occurred while processing your request",
                "details": str(e)
            }
            return f"\n\n[ERROR]{error_response}"

    def _get_vector_store(self, files: List[str]) -> Chroma:
        # Implementation of vector store creation
        logger.info(f"Processing files: {files}")
        try:
            # Process files and create vector store
            text = file_processor.process_files(files)
            logger.info("Creating text splitter")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_text(text)
            logger.info(f"Split text into {len(texts)} chunks")
            
            logger.info("Creating vector store")
            # Create a unique collection name based on timestamp to avoid dimension conflicts
            import time
            collection_name = f"collection_{int(time.time())}"
            persist_directory = "chroma_db"
            vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            logger.info("Vector store created")
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def remove_vector_store(self):
        try:
            # Get all collections in the database
            collections = self.chroma_client.list_collections()
            for collection in collections:
                # Delete each collection
                self.chroma_client.delete_collection(collection.name)
            logger.info("All collections deleted")
        except Exception as e:
            logger.error(f"Error deleting collections: {str(e)}")
            raise

    def _get_encoding(self, model: str):
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding if model is not found
            logger.warning(f"Model {model} not found in tiktoken, using cl100k_base encoding")
            return tiktoken.get_encoding("cl100k_base")

    def track_actual_token_usage(self, messages: List[dict], response: str, model: str) -> dict:
        """
        Track actual token usage for both input and output.
        This provides a more accurate count than relying on provider metadata.
        """
        try:
            encoding = self._get_encoding(model)
            
            # Count input tokens
            prompt_tokens = self.estimate_tokens(messages, model)
            
            # Count output tokens
            completion_tokens = len(encoding.encode(response))
            
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        except Exception as e:
            logger.error(f"Error tracking token usage: {str(e)}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def estimate_tokens(self, messages: List[dict], model: str) -> int:
        """
        Estimate the number of tokens for a list of messages.
        This follows OpenAI's token counting rules for chat completions.
        """
        encoding = self._get_encoding(model)
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def estimate_response_tokens(self, prompt_tokens: int) -> int:
        """
        Estimate the number of tokens in the response based on the prompt tokens.
        This is a rough estimation as the actual response length can vary.
        """
        # A common rule of thumb is that responses are typically 1.5-2x the length of the prompt
        # We'll use 1.5x as a conservative estimate
        return int(prompt_tokens * 1.5)

    def estimate_total_tokens(self, messages: List[dict], system_template: str, model: str) -> dict:
        try:
            """
            Estimate total token usage including both prompt and response.
            Returns a dictionary with estimated token counts.
            """
            encoding = self._get_encoding(model)
            print("encoding", system_template)
            
            prompt_tokens = self.estimate_tokens(messages, model) + len(encoding.encode(system_template))
            print("prompt_tokens", prompt_tokens)
            response_tokens = self.estimate_response_tokens(prompt_tokens)
            print("response_tokens", response_tokens)
            total_tokens = prompt_tokens + response_tokens
            print("total_tokens", total_tokens)

            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": response_tokens,
                "total_tokens": total_tokens
            }
        except Exception as e:
            logger.error(f"Error estimating total tokens: {str(e)}")
            raise

chat_service = ChatService() 