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
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
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
        system_prompt = "" if provider != "edith" else db.get_system_prompt()
        messages = []
        
        logger.info(f"Processing chat history for provider: {provider}")
        logger.info(f"Chat history length: {len(chat_history)}")
        
        # For Anthropic, we don't include system message in the messages array
        if provider.lower() != "anthropic":
            messages.append({"role": "system", "content": system_prompt})
            logger.info("Added system message")
            
        # For DeepSeek, ensure first message is from user
        if provider.lower() == "deepseek":
            # First, filter out any empty or invalid messages
            valid_history = [chat for chat in chat_history if chat.prompt or chat.response]
            logger.info(f"Valid history length: {len(valid_history)}")
            
            if not valid_history:
                logger.warning("No valid messages in chat history")
                return messages, system_prompt
                
            # Ensure we start with a user message
            first_chat = valid_history[0]
            if not first_chat.prompt:
                logger.warning("First message is not a user message, skipping chat history")
                return messages, system_prompt
                
            # Add the first user message
            messages.append({"role": "user", "content": first_chat.prompt})
            logger.info("Added first user message")
            
            # Add its response if it exists
            if first_chat.response:
                messages.append({"role": "assistant", "content": first_chat.response})
                logger.info("Added first assistant response")
            
            # Add remaining messages in pairs
            for chat in valid_history[1:]:
                if chat.prompt:
                    messages.append({"role": "user", "content": chat.prompt})
                    logger.info("Added user message")
                if chat.response:
                    messages.append({"role": "assistant", "content": chat.response})
                    logger.info("Added assistant response")
        else:
            # Original behavior for other providers
            for chat in chat_history:
                if chat.prompt:
                    messages.append({"role": "user", "content": chat.prompt})
                if chat.response:
                    messages.append({"role": "assistant", "content": chat.response})
        
        logger.info(f"Final messages count: {len(messages)}")
        logger.info(f"Message roles: {[msg['role'] for msg in messages]}")
        
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
                max_tokens=2000,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                stream_usage=isStream
            )
        elif ai_config.provider.lower() == "deepseek":
            return ChatDeepSeek(
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                max_tokens=2000,
                api_key=settings.DEEPSEEK_API_KEY,
                stream_usage=isStream,
            )
        elif ai_config.provider.lower() == "google":
            return ChatGoogleGenerativeAI(
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                max_tokens=2000,
                google_api_key=settings.GOOGLE_API_KEY,
                stream_usage=isStream
            )
        elif ai_config.provider.lower() == "xai":
            return ChatXAI(   
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                max_tokens=2000,
                xai_api_key=settings.XAI_API_KEY,
                stream_usage=isStream
            )
        elif ai_config.provider.lower() == "ollama":
            return ChatOllama(
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                max_tokens=2000,
                base_url=settings.OLLAMA_BASE_URL,
                stream_usage=isStream
            )
        elif ai_config.provider.lower() == "mistralai":
            return ChatMistralAI(
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                max_tokens=2000,
                mistral_api_key=settings.MISTRAL_API_KEY,
                stream_usage=isStream
            )
        elif ai_config.provider.lower() == "cerebras" or ai_config.provider.lower() == "edith":
            return ChatCerebras(
                temperature=0.7,
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model="llama-3.3-70b" if ai_config.provider.lower() == "edith" else ai_config.model,
                max_tokens=2000,
                cerebras_api_key=settings.CEREBRAS_API_KEY,
                stream_usage=isStream
            )
        elif ai_config.provider.lower() == "openrouter":
            return ChatOpenAI(
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                max_tokens=2000,
                openai_api_key=settings.OPENROUTER_API_KEY,
                stream_usage=isStream,
                base_url="https://openrouter.ai/api/v1"
            )
        else:  # Default to OpenAI
            return ChatOpenAI(
                streaming=isStream,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                max_tokens=2000,
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
        learningPrompt: str,
    ) -> AsyncGenerator[str, None]:
        logger.info(f"Generating response for query: {query}")
        full_response = ""
        points = 0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        outputTime = datetime.now()
        has_started_reasoning = False  # Track if we've started collecting reasoning

        try:
            # Initialize user_point with email
            await user_point.initialize(email)
            
            ai_config = db.get_ai_config(model)
            print("ai_config", ai_config)
            if not ai_config:
                yield "Error: Invalid AI configuration"
                return
            
            llm = self._get_llm(ai_config)

            if files:
                print("Using RAG with files")
                image_files, text_files = file_processor.identify_files(files)
                vector_store = self._get_vector_store(text_files) if text_files else None
                
                # Get messages for token estimation
                messages, system_prompt = self.get_chat_messages(chat_history, ai_config.provider)
                
                # Handle multimodal input (text + images)
                if ai_config.imageSupport and image_files:
                    multimodal_message = self.create_multimodal_message(query, image_files, ai_config.provider)
                    messages.append(multimodal_message)
                else:
                    messages.append({"role": "user", "content": query})

                if vector_store:
                    system_template = system_prompt + "\n\nPrevious conversation:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}"
                    
                    # Get relevant context from files for token estimation
                    retriever = vector_store.as_retriever()
                    docs = retriever.get_relevant_documents(query)
                    context = "\n".join([doc.page_content for doc in docs])
                else:
                    system_template = system_prompt
                    context = ""
                
                # Estimate tokens before making the API call
                estimated_tokens = self.estimate_total_tokens(messages, system_template, "llama3.1-8b" if ai_config.provider.lower() == "edith" else ai_config.model, context)
                estimated_points = self.get_points(estimated_tokens["prompt_tokens"], estimated_tokens["completion_tokens"], ai_config)
                print(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")
                
                check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points, ai_config)
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
                    *[HumanMessagePromptTemplate.from_template(self._extract_text_content(msg["content"])) if msg["role"] == "user" 
                      else AIMessagePromptTemplate.from_template(msg["content"]) 
                      for msg in messages[:-1] if self._has_content(msg)],  # Only include messages with content
                    HumanMessagePromptTemplate.from_template("{question}")  # Last message is the current question
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
                    logger.info(f"Event stream: {event}")
                    
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
                    elif event["event"] == "on_chat_model_stream":
                        # Handle reasoning content at the model stream level
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "additional_kwargs") and "reasoning_content" in chunk.additional_kwargs:
                            reasoning = chunk.additional_kwargs["reasoning_content"]
                            if reasoning:
                                if not has_started_reasoning:
                                    yield "<think>"
                                    has_started_reasoning = True
                                    full_response += "<think>"
                                yield reasoning
                                full_response += reasoning
                                continue
                    elif event["event"] == "on_chain_stream" and event["name"] == "RunnableSequence":
                        try:
                            chunk = event["data"]["chunk"]
                            if chunk is None:
                                continue
                            # If we were collecting reasoning and now we're getting actual content, close the think tag
                            if has_started_reasoning and chunk_str != "":
                                yield "</think>\n"
                                has_started_reasoning = False
                                full_response += "</think>\n"

                            chunk_str = str(chunk) if not isinstance(chunk, str) else chunk
                            full_response += chunk_str
                            yield chunk_str
                        except Exception as e:
                            logger.error(f"Error processing chunk: {str(e)}")
                            continue
                        await asyncio.sleep(0.01)
                
                points = self.get_points(token_usage["prompt_tokens"], token_usage["completion_tokens"], ai_config)
                yield f"\n\n[POINTS]{points}"
                
            else:
                print(f"Using direct {ai_config.provider} completion")
                messages, system_prompt = self.get_chat_messages(chat_history, ai_config.provider)
                messages.append({"role": "user", "content": learningPrompt if chatType == 1 else query})
                
                system_template = system_prompt
                estimated_tokens = self.estimate_total_tokens(messages, system_template, "llama3.1-8b" if ai_config.provider.lower() == "edith" else ai_config.model)
                estimated_points = self.get_points(estimated_tokens["prompt_tokens"], estimated_tokens["completion_tokens"], ai_config)
                print(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")
                
                check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points, ai_config)
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
                    *[HumanMessagePromptTemplate.from_template(self._extract_text_content(msg["content"])) if msg["role"] == "user" 
                      else AIMessagePromptTemplate.from_template(msg["content"]) 
                      for msg in messages[:-1] if self._has_content(msg)],  # Only include messages with content
                    HumanMessagePromptTemplate.from_template("{question}")  # Last message is the current question
                ])
                
                chain = (
                    {"question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                async for event in chain.astream_events(query):
                    logger.info(f"Event stream: {event}")
                    
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
                    elif event["event"] == "on_chat_model_stream":
                        # Handle reasoning content at the model stream level
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "additional_kwargs") and "reasoning_content" in chunk.additional_kwargs:
                            reasoning = chunk.additional_kwargs["reasoning_content"]
                            if reasoning:
                                if not has_started_reasoning:
                                    yield "<think>"
                                    full_response += "<think>"
                                    has_started_reasoning = True
                                yield reasoning
                                full_response += reasoning
                                continue
                    elif event["event"] == "on_chain_stream" and event["name"] == "RunnableSequence":
                        try:
                            chunk = event["data"]["chunk"]
                            if chunk is None:
                                continue
                            chunk_str = str(chunk) if not isinstance(chunk, str) else chunk
                            if has_started_reasoning and chunk_str != "":
                                yield "</think>\n"
                                full_response += "</think>\n"
                                has_started_reasoning = False
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
                    "points": points,
                    "count": 1
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
            llm = self._get_llm(ai_config, False)

            if files:
                print("Using RAG with files")
                image_files, text_files = file_processor.identify_files(files)
                vector_store = self._get_vector_store(text_files) if text_files else None
                
                # Get messages for token estimation
                messages, system_prompt = self.get_chat_messages(chat_history, ai_config.provider)
                
                # Handle multimodal input (text + images)
                if ai_config.imageSupport and image_files:
                    multimodal_message = self.create_multimodal_message(query, image_files, ai_config.provider)
                    messages.append(multimodal_message)
                else:
                    messages.append({"role": "user", "content": query})
                
                if vector_store:
                    system_template = system_prompt + "\n\nPrevious conversation:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}"

                    # Get relevant context from files for token estimation
                    retriever = vector_store.as_retriever()
                    docs = retriever.get_relevant_documents(query)
                    context = "\n".join([doc.page_content for doc in docs])
                else:
                    system_template = system_prompt
                    context = ""
                
                # Estimate tokens before making the API call
                estimated_tokens = self.estimate_total_tokens(messages, system_template, "llama3.1-8b" if ai_config.provider.lower() == "edith" else ai_config.model, context)
                estimated_points = self.get_points(estimated_tokens["prompt_tokens"], estimated_tokens["completion_tokens"], ai_config)
                print(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")
                
                check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points, ai_config)
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
                    *[HumanMessagePromptTemplate.from_template(self._extract_text_content(msg["content"])) if msg["role"] == "user" 
                      else AIMessagePromptTemplate.from_template(msg["content"]) 
                      for msg in messages[:-1] if self._has_content(msg)],  # Only include messages with content
                    HumanMessagePromptTemplate.from_template("{question}")  # Last message is the current question
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

            else:
                print(f"Using direct {ai_config.provider} completion")
                messages, system_prompt = self.get_chat_messages(chat_history, ai_config.provider)
                messages.append({"role": "user", "content": query})
                
                system_template = system_prompt
                estimated_tokens = self.estimate_total_tokens(messages, system_template, "llama3.1-8b" if ai_config.provider.lower() == "edith" else ai_config.model)
                estimated_points = self.get_points(estimated_tokens["prompt_tokens"], estimated_tokens["completion_tokens"], ai_config)
                print(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")
                
                check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points, ai_config)
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
                    *[HumanMessagePromptTemplate.from_template(self._extract_text_content(msg["content"])) if msg["role"] == "user" 
                      else AIMessagePromptTemplate.from_template(msg["content"]) 
                      for msg in messages[:-1] if self._has_content(msg)],  # Only include messages with content
                    HumanMessagePromptTemplate.from_template("{question}")  # Last message is the current question
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
            
            # # Remove vector store after stream is fully completed
            # if files:
            #     self.remove_vector_store()

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
                    "points": points,
                    "count": 1
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
            
            # Process files if they exist
            if files:
                print("Using RAG with files")
                image_files, text_files = file_processor.identify_files(files)
                vector_store = self._get_vector_store(text_files) if text_files else None
                
                # Handle multimodal input (text + images) for context
                if ai_config.imageSupport and image_files:
                    multimodal_message = self.create_multimodal_message(query, image_files, ai_config.provider)
                    messages.append(multimodal_message)
                else:
                    messages.append({"role": "user", "content": query})
                
                if vector_store:
                    # Get relevant context from files
                    retriever = vector_store.as_retriever()
                    docs = retriever.get_relevant_documents(query)
                    context = "\n".join([doc.page_content for doc in docs])
                    # Enhance the prompt with context
                    enhanced_query = f"Context from files:\n{context}\n\nGenerate image based on this context and the following description: {query}"
                else:
                    enhanced_query = query
            else:
                messages.append({"role": "user", "content": query})
                enhanced_query = query

            # Estimate tokens for the prompt
            estimated_tokens = self.estimate_image_tokens(enhanced_query)  # Use image-specific token estimation
            estimated_points = self.get_points(estimated_tokens["prompt_tokens"], estimated_tokens["completion_tokens"], ai_config)
            print(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")

            # Check if user has enough points
            check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points, ai_config)
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

            # # Remove vector store after stream is fully completed
            # if files:
            #     self.remove_vector_store()

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
                    "points": points,
                    "count": 1
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

            # Process files if they exist
            if files:
                print("Using RAG with files")
                image_files, text_files = file_processor.identify_files(files)
                vector_store = self._get_vector_store(text_files) if text_files else None
                
                # Handle multimodal input (text + images) for context
                if ai_config.imageSupport and image_files:
                    multimodal_message = self.create_multimodal_message(query, image_files, ai_config.provider)
                    messages.append(multimodal_message)
                else:
                    messages.append({"role": "user", "content": query})
                
                if vector_store:
                    # Get relevant context from files
                    retriever = vector_store.as_retriever()
                    docs = retriever.get_relevant_documents(query)
                    context = "\n".join([doc.page_content for doc in docs])
                    # Enhance the prompt with context
                    enhanced_query = f"Context from files:\n{context}\n\nGenerate audio based on this context and the following text: {query}"
                else:
                    enhanced_query = query
            else:
                messages.append({"role": "user", "content": query})
                enhanced_query = query

            # Estimate tokens for audio generation
            estimated_tokens = self.estimate_audio_tokens(enhanced_query)
            estimated_points = self.get_points(estimated_tokens["prompt_tokens"], estimated_tokens["completion_tokens"], ai_config)
            print(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")

            # Check if user has enough points
            check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points, ai_config)
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

            # # Remove vector store after stream is fully completed
            # if files:
            #     self.remove_vector_store()

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
                    "points": points,
                    "count": 1
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

    def _cleanup_old_collections(self, max_age_hours: int = 24):
        """
        Clean up old collections from Chroma DB.
        Args:
            max_age_hours: Maximum age of collections in hours before they are deleted
        """
        try:
            from chromadb import Client
            import time
            from datetime import datetime, timedelta

            client = Client()
            collections = client.list_collections()
            current_time = datetime.now()
            
            for collection in collections:
                # Extract timestamp from collection name
                try:
                    timestamp = int(collection.name.split('_')[1])
                    collection_time = datetime.fromtimestamp(timestamp)
                    age = current_time - collection_time
                    
                    # Delete if collection is older than max_age_hours
                    if age > timedelta(hours=max_age_hours):
                        client.delete_collection(collection.name)
                        logger.info(f"Deleted old collection: {collection.name}")
                except (ValueError, IndexError):
                    # If collection name doesn't match expected format, skip it
                    continue
                    
        except Exception as e:
            logger.error(f"Error cleaning up old collections: {str(e)}")
            # Don't raise the exception as this is a cleanup operation

    def _get_vector_store(self, files: List[str]) -> Chroma:
        """
        Create or get a vector store for the given files.
        Returns a Chroma vector store instance.
        """
        logger.info(f"Processing files: {files}")
        try:
            # Clean up old collections first
            self._cleanup_old_collections()
            
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
            # Create a unique collection name based on timestamp
            import time
            collection_name = f"collection_{int(time.time())}"
            persist_directory = "chroma_db"
            
            try:
                # Try to create new vector store
                vector_store = Chroma.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    persist_directory=persist_directory,
                    collection_name=collection_name
                )
                logger.info(f"Created new vector store with collection: {collection_name}")
            except Exception as e:
                logger.error(f"Error creating new vector store: {str(e)}")
                # If creation fails, try to use existing collection
                from chromadb import Client
                client = Client()
                existing_collections = client.list_collections()
                
                if existing_collections:
                    # Use the most recent collection
                    latest_collection = max(existing_collections, 
                                         key=lambda x: int(x.name.split('_')[1]))
                    vector_store = Chroma(
                        client=client,
                        collection_name=latest_collection.name,
                        embedding_function=self.embeddings,
                        persist_directory=persist_directory
                    )
                    logger.info(f"Using existing collection: {latest_collection.name}")
                else:
                    raise Exception("No existing collections available and failed to create new one")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error in _get_vector_store: {str(e)}")
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
        This follows OpenAI's token counting rules for chat completions and includes vision tokens.
        """
        encoding = self._get_encoding(model)
        num_tokens = 0
        
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            
            for key, value in message.items():
                if key == "content":
                    if isinstance(value, str):
                        # Regular text content
                        num_tokens += len(encoding.encode(value))
                    elif isinstance(value, list):
                        # Multimodal content
                        for content_item in value:
                            if isinstance(content_item, dict):
                                if content_item.get("type") == "text":
                                    text = content_item.get("text", "")
                                    num_tokens += len(encoding.encode(text))
                                elif content_item.get("type") == "image_url":
                                    # Estimate vision tokens
                                    # OpenAI vision models use approximately 85-170 tokens per image
                                    # depending on detail level and image size
                                    detail = content_item.get("image_url", {}).get("detail", "auto")
                                    if detail == "low":
                                        num_tokens += 85
                                    else:  # high or auto
                                        num_tokens += 170
                                elif content_item.get("type") == "image":
                                    # Anthropic/Google image format
                                    num_tokens += 170  # Conservative estimate
                    elif isinstance(value, dict):
                        # Single content object
                        if "text" in value:
                            num_tokens += len(encoding.encode(value["text"]))
                else:
                    # Other message fields (role, name, etc.)
                    if isinstance(value, str):
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
        max_tokens = 2000
        return min(int(prompt_tokens * 1.5), max_tokens)

    def estimate_total_tokens(self, messages: List[dict], system_template: str, model: str, context = "") -> dict:
        try:
            """
            Estimate total token usage including both prompt and response.
            Returns a dictionary with estimated token counts.
            """
            encoding = self._get_encoding(model)
            print("encoding", system_template)
            
            prompt_tokens = self.estimate_tokens(messages, model) + len(encoding.encode(system_template)) + len(encoding.encode(context))
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

    def indentify_files(self, files: List[str]) -> List[str]:
        """
        Identify the type of files and return the image files and the text files.
        """
        return file_processor.identify_files(files)

    def format_image_content(self, image_files: List[str], provider: str) -> List[dict]:
        """
        Format image files for different AI providers.
        Returns a list of content objects that can be added to messages.
        """
        formatted_content = []
        
        for image_url in image_files:
            if provider.lower() == "openai":
                # OpenAI format for vision models
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url if image_url.startswith('http') else f"{settings.AWS_CDN_URL}/{image_url}",
                        "detail": "high"  # Can be "low", "high", or "auto"
                    }
                })
            elif provider.lower() == "anthropic":
                # Anthropic Claude format for vision
                # Note: Anthropic requires base64 encoded images
                try:
                    import requests
                    import base64
                    
                    # Download the image
                    full_url = image_url if image_url.startswith('http') else f"{settings.AWS_CDN_URL}/{image_url}"
                    response = requests.get(full_url)
                    response.raise_for_status()
                    
                    # Convert to base64
                    image_base64 = base64.b64encode(response.content).decode('utf-8')
                    
                    # Determine media type
                    if image_url.lower().endswith('.png'):
                        media_type = "image/png"
                    elif image_url.lower().endswith(('.jpg', '.jpeg')):
                        media_type = "image/jpeg"
                    elif image_url.lower().endswith('.gif'):
                        media_type = "image/gif"
                    elif image_url.lower().endswith('.webp'):
                        media_type = "image/webp"
                    else:
                        media_type = "image/jpeg"  # Default fallback
                    
                    formatted_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64
                        }
                    })
                except Exception as e:
                    logger.error(f"Error processing image for Anthropic: {str(e)}")
                    continue
            elif provider.lower() == "google":
                # Google Gemini format
                try:
                    import requests
                    import base64
                    
                    # Download the image
                    full_url = image_url if image_url.startswith('http') else f"{settings.AWS_CDN_URL}/{image_url}"
                    response = requests.get(full_url)
                    response.raise_for_status()
                    
                    # Convert to base64
                    image_base64 = base64.b64encode(response.content).decode('utf-8')
                    
                    # Determine mime type
                    if image_url.lower().endswith('.png'):
                        mime_type = "image/png"
                    elif image_url.lower().endswith(('.jpg', '.jpeg')):
                        mime_type = "image/jpeg"
                    elif image_url.lower().endswith('.gif'):
                        mime_type = "image/gif"
                    elif image_url.lower().endswith('.webp'):
                        mime_type = "image/webp"
                    else:
                        mime_type = "image/jpeg"  # Default fallback
                    
                    formatted_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    })
                except Exception as e:
                    logger.error(f"Error processing image for Google: {str(e)}")
                    continue
            else:
                # Default format (similar to OpenAI)
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url if image_url.startswith('http') else f"{settings.AWS_CDN_URL}/{image_url}",
                        "detail": "high"
                    }
                })
        
        return formatted_content

    def create_multimodal_message(self, text_content: str, image_files: List[str], provider: str) -> dict:
        """
        Create a multimodal message with both text and images.
        """
        if not image_files:
            return {"role": "user", "content": text_content}
        
        if provider.lower() == "openai":
            # OpenAI multimodal format
            content = [
                {"type": "text", "text": text_content}
            ]
            content.extend(self.format_image_content(image_files, provider))
            return {"role": "user", "content": content}
        
        elif provider.lower() == "anthropic":
            # Anthropic multimodal format
            content = [
                {"type": "text", "text": text_content}
            ]
            content.extend(self.format_image_content(image_files, provider))
            return {"role": "user", "content": content}
        
        elif provider.lower() == "google":
            # Google Gemini multimodal format
            content = [
                {"type": "text", "text": text_content}
            ]
            content.extend(self.format_image_content(image_files, provider))
            return {"role": "user", "content": content}
        
        elif provider.lower() == "openrouter":
            # OpenRouter multimodal format
            content = [
                {"type": "text", "text": text_content}
            ]
            content.extend(self.format_image_content(image_files, provider))
            return {"role": "user", "content": content}
        
        else:
            # For providers that don't support vision, just return text
            logger.warning(f"Provider {provider} doesn't support vision. Images will be ignored.")
            return {"role": "user", "content": text_content}

    def _extract_text_content(self, content):
        """Extract text content from message content, handling both string and multimodal formats."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Multimodal content - find text parts
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return " ".join(text_parts)
        elif isinstance(content, dict) and "text" in content:
            return content["text"]
        else:
            logger.warning(f"Unsupported content format: {type(content)}")
            return ""

    def _has_content(self, message):
        """Check if a message has content."""
        if not isinstance(message, dict):
            return False
        
        content = message.get("content")
        if not content:
            return False
            
        if isinstance(content, str):
            return bool(content.strip())
        elif isinstance(content, list):
            # Check if any item in the list has meaningful content
            return any(
                (isinstance(item, dict) and item.get("text", "").strip()) or
                (isinstance(item, dict) and item.get("type") in ["image_url", "image"])
                for item in content
            )
        elif isinstance(content, dict):
            return bool(content.get("text", "").strip())
        
        return False

    def get_supported_image_info(self) -> dict:
        """
        Returns information about supported image formats and usage guidelines.
        """
        return {
            "supported_formats": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".ico", ".webp"],
            "max_file_size": "20MB (recommended)",
            "usage_guidelines": {
                "openai": {
                    "description": "Supports vision with GPT-4V and GPT-4o models",
                    "detail_levels": ["low", "high", "auto"],
                    "token_cost": "85 tokens (low detail) to 170 tokens (high detail) per image"
                },
                "anthropic": {
                    "description": "Supports vision with Claude 3 models",
                    "formats": ["base64 encoded images"],
                    "token_cost": "~170 tokens per image (estimated)"
                },
                "google": {
                    "description": "Supports vision with Gemini models",
                    "formats": ["base64 encoded images"],
                    "token_cost": "~170 tokens per image (estimated)"
                }
            },
            "tips": [
                "Images are automatically detected by file extension",
                "High-resolution images provide better analysis but cost more tokens",
                "Combine images with text prompts for best results",
                "Multiple images can be processed in a single request"
            ]
        }

chat_service = ChatService() 