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
from langchain_community.embeddings import HuggingFaceEmbeddings
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

logger = logging.getLogger(__name__)

class NoPointsAvailableException(Exception):
    pass

class ChatService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_stores = {}
        self.encoding = None  # Will be set based on the model being used

    def get_chat_messages(self, chat_history: List[IRouterChatLog], provider: str = "openai") -> List[dict]:
        system_prompt = db.get_system_prompt()
        messages = []
        
        # For Anthropic, we don't include system message in the messages array
        if provider.lower() != "anthropic":
            messages.append({"role": "system", "content": system_prompt})
            
        for chat in chat_history:
            if chat.prompt:
                messages.append({"role": "user", "content": chat.prompt})
            if chat.response:
                messages.append({"role": "assistant", "content": chat.response})
                
        return messages, system_prompt if provider.lower() == "anthropic" else None

    def get_points(self, inputToken: int, outputToken: int, ai_config: AiConfig) -> float:
        return (inputToken * ai_config.inputCost + outputToken * ai_config.outputCost) * ai_config.multiplier / 0.001

    def _get_llm(self, ai_config: AiConfig):
        if ai_config.provider.lower() == "anthropic":
            return ChatAnthropic(
                temperature=0.7,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                stream_usage=True
            )
        elif ai_config.provider.lower() == "deepseek":
            return ChatDeepSeek(
                temperature=0.7,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                deepseek_api_key=settings.DEEPSEEK_API_KEY,
                stream_usage=True
            )
        elif ai_config.provider.lower() == "google":
            return ChatGoogleGenerativeAI(
                temperature=0.7,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                google_api_key=settings.GOOGLE_API_KEY,
                stream_usage=True
            )
        elif ai_config.provider.lower() == "xai":
            return ChatXAI(   temperature=0.7,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                xai_api_key=settings.XAI_API_KEY,
                stream_usage=True
            )
        elif ai_config.provider.lower() == "ollama":
            return ChatOllama(
                temperature=0.7,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                base_url=settings.OLLAMA_BASE_URL,
                stream_usage=True
            )
        elif ai_config.provider.lower() == "mistralai":
            return ChatMistralAI(
                temperature=0.7,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                mistral_api_key=settings.MISTRAL_API_KEY,
                stream_usage=True
            )
        else:  # Default to OpenAI
            return ChatOpenAI(
                temperature=0.7,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                model=ai_config.model,
                openai_api_key=settings.OPENAI_API_KEY,
                stream_usage=True
            )

    async def generate_stream_response(
        self,
        query: str,
        files: List[str],
        chat_history: List[IRouterChatLog],
        model: str,
        email: str,
        sessionId: str,
        reGenerate: bool
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
                logger.info("Using RAG with files")
                vector_store = self._get_vector_store(files)
                
                # Get messages for token estimation
                messages, system_prompt = self.get_chat_messages(chat_history, ai_config.provider)
                messages.append({"role": "user", "content": query})
                
                system_template = (system_prompt or db.get_system_prompt()) + "\n\nPrevious conversation:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}"
                
                # Estimate tokens before making the API call
                estimated_tokens = self.estimate_total_tokens(messages, system_template, model)
                estimated_points = self.get_points(estimated_tokens["prompt_tokens"], 0, ai_config)
                logger.info(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")
                
                check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points)
                if not check_user_available_to_chat:
                    raise NoPointsAvailableException("User has no available points")
                
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
                        chunk = event["data"]["chunk"]
                        if isinstance(chunk, str):
                            full_response += chunk
                            yield chunk
                        else:
                            full_response += str(chunk)
                            yield str(chunk)
                        await asyncio.sleep(0.01)
                
                points = self.get_points(token_usage["prompt_tokens"], token_usage["completion_tokens"], ai_config)
                yield f"\n\n[POINTS]{points}"
                
            else:
                logger.info(f"Using direct {ai_config.provider} completion")
                messages, system_prompt = self.get_chat_messages(chat_history, ai_config.provider)
                messages.append({"role": "user", "content": query})
                
                # Estimate tokens before making the API call
                system_template = system_prompt or db.get_system_prompt()
                estimated_tokens = self.estimate_total_tokens(messages, system_template, model)
                estimated_points = self.get_points(estimated_tokens["prompt_tokens"], 0, ai_config)
                logger.info(f"Estimated token usage: {estimated_tokens}, Estimated points: {estimated_points}")
                
                check_user_available_to_chat = await user_point.check_user_available_to_chat(estimated_points)
                if not check_user_available_to_chat:
                    raise NoPointsAvailableException("User has no available points")
                
                llm = self._get_llm(ai_config)
                
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
                        chunk = event["data"]["chunk"]
                        if isinstance(chunk, str):
                            full_response += chunk
                            yield chunk
                        else:
                            full_response += str(chunk)
                            yield str(chunk)
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
                    "model": model,
                    "response": full_response,
                    "timestamp": datetime.now(),
                    "inputToken": token_usage["prompt_tokens"],
                    "outputToken": token_usage["completion_tokens"],
                    "outputTime": outputTime,
                    "fileUrls": files,
                    "points": points
                }
            })
            await user_point.save_user_points(points)
        except NoPointsAvailableException as e:
            logger.error(f"Error in generate_stream_response: {str(e)}")
            raise HTTPException(status_code=429, detail=str(e))
        except Exception as e:
            logger.error(f"Error in generate_stream_response: {str(e)}")
            yield f"Error: {str(e)}"

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
            # Create a temporary directory for Chroma
            persist_directory = "chroma_db"
            vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            logger.info("Vector store created")
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
        pass

    def _get_encoding(self, model: str):
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding if model is not found
            logger.warning(f"Model {model} not found in tiktoken, using cl100k_base encoding")
            return tiktoken.get_encoding("cl100k_base")

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
        """
        Estimate total token usage including both prompt and response.
        Returns a dictionary with estimated token counts.
        """
        encoding = self._get_encoding(model)
        prompt_tokens = self.estimate_tokens(messages, model) + len(encoding.encode(system_template))
        response_tokens = self.estimate_response_tokens(prompt_tokens)
        total_tokens = prompt_tokens + response_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": response_tokens,
            "total_tokens": total_tokens
        }

chat_service = ChatService() 