from typing import AsyncGenerator, List, Optional
import asyncio
from datetime import datetime
import logging
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from ..config.settings import settings
from ..core.database import db
from ..models.chat import ChatRequest, IRouterChatLog, AiConfig
from ..utils.file_processor import process_files

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.vector_stores = {}

    def get_chat_messages(self, chat_history: List[IRouterChatLog]) -> List[dict]:
        messages = [{"role": "system", "content": db.get_system_prompt()}]
        for chat in chat_history:
            if chat.prompt:
                messages.append({"role": "user", "content": chat.prompt})
            if chat.response:
                messages.append({"role": "assistant", "content": chat.response})
        return messages

    def get_points(self, inputToken: int, outputToken: int, ai_config: AiConfig) -> float:
        return (inputToken * ai_config.inputCost + outputToken * ai_config.outputCost) * ai_config.multiplier / 0.001

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
            ai_config = db.get_ai_config(model)
            if not ai_config:
                yield "Error: Invalid AI configuration"
                return

            if files:
                logger.info("Using RAG with files")
                vector_store = self._get_vector_store(files)
                
                llm = ChatOpenAI(
                    temperature=0.7,
                    streaming=True,
                    callbacks=[StreamingStdOutCallbackHandler()],
                    model=settings.DEFAULT_MODEL
                )
                
                system_template = db.get_system_prompt() + "\n\nPrevious conversation:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}"
                
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
                logger.info("Using direct OpenAI completion")
                messages = self.get_chat_messages(chat_history)
                messages.append({"role": "user", "content": query})
                
                stream = self.client.chat.completions.create(
                    model=settings.DEFAULT_MODEL,
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    stream_options={"include_usage": True}
                )
                
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        if isinstance(content, str):
                            full_response += content
                            yield content
                        else:
                            full_response += str(content)
                            yield str(content)
                        await asyncio.sleep(0.01)
                    
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        token_usage = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens
                        }
                        points = self.get_points(token_usage["prompt_tokens"], token_usage["completion_tokens"], ai_config)
                        logger.info(f"Token usage for direct completion: {token_usage}, Cost: ${points}")
                
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
            
        except Exception as e:
            logger.error(f"Error in generate_stream_response: {str(e)}")
            yield f"Error: {str(e)}"

    def _get_vector_store(self, files: List[str]) -> Chroma:
        # Implementation of vector store creation
        logger.info(f"Processing files: {files}")
        try:
            # Process files and create vector store
            text = process_files(files)
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

chat_service = ChatService() 