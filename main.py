from typing import Union, AsyncGenerator, List
import asyncio
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import os
from dotenv import load_dotenv
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import tempfile
from pathlib import Path
import docx
from PyPDF2 import PdfReader
import io
import logging
import urllib3
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ChatRequest(BaseModel):
    sessionId: str
    prompt: str
    model: str = "gpt-3.5-turbo"
    reGenerate: bool = False
    files: List[str] = []

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model_name = 'text-embedding-ada-002'

embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Store vector stores in memory (in production, use a proper database)
vector_stores = {}

def download_file(url: str) -> bytes:
    logger.info(f"Downloading file from URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        logger.info(f"Successfully downloaded file from {url}")
        return response.content
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        raise

def process_pdf(content: bytes) -> str:
    pdf_reader = PdfReader(io.BytesIO(content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_docx(content: bytes) -> str:
    doc = docx.Document(io.BytesIO(content))
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def process_txt(content: bytes) -> str:
    return content.decode('utf-8')

def process_files(files: List[str]) -> str:
    logger.info(f"Processing files: {files}")
    all_text = ""
    for file_url in files:
        try:
            content = download_file(file_url)
            file_extension = Path(file_url).suffix.lower()
            logger.info(f"Processing file with extension: {file_extension}")
            
            if file_extension == '.pdf':
                text = process_pdf(content)
            elif file_extension == '.docx':
                text = process_docx(content)
            elif file_extension == '.txt':
                text = process_txt(content)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            all_text += text + "\n\n"
            logger.info(f"Successfully processed file: {file_url}")
        except Exception as e:
            logger.error(f"Error processing file {file_url}: {str(e)}")
            raise
    return all_text

def get_vector_store(session_id: str, files: List[str]) -> FAISS:
    logger.info(f"Getting vector store for session: {session_id}")
    # if session_id in vector_stores:
    #     logger.info("Using cached vector store")
    #     return vector_stores[session_id]
    
    try:
        # Process files and create vector store
        text = process_files(files)
        logger.info("Creating text splitter")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_text(text)
        logger.info(f"Split text into {len(texts)} chunks")
        
        logger.info("Creating vector store")
        vector_store = FAISS.from_texts(texts, embeddings)
        vector_stores[session_id] = vector_store
        logger.info("Vector store created and cached")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

async def generate_stream_response(query: str, session_id: str, files: List[str]) -> AsyncGenerator[str, None]:
    logger.info(f"Generating response for query: {query}")
    try:
        if files:
            logger.info("Using RAG with files")
            vector_store = get_vector_store(session_id, files)
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            logger.info("Creating QA chain")
            # Create a streaming LLM
            llm = ChatOpenAI(
                temperature=0.7,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(),
                memory=memory,
                return_source_documents=True,
                output_key="answer"
            )
            
            logger.info("Running QA chain")
            response = qa_chain.invoke({"question": query})
            answer = response["answer"]
            
            # Stream the answer in smaller chunks
            chunk_size = 10
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.01)
                
        else:
            logger.info("Using direct OpenAI completion")
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}],
                stream=True,
                temperature=0.7,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield content
                    await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"Error in generate_stream_response: {str(e)}")
        yield f"Error: {str(e)}"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    logger.info(f"Received chat request - Session: {request.sessionId}, Prompt: {request.prompt}")
    logger.info(f"Files: {request.files}")
    return StreamingResponse(
        generate_stream_response(request.prompt, request.sessionId, request.files),
        media_type="text/event-stream"
    )
    