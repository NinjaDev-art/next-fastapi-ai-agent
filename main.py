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
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
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

def get_vector_store(files: List[str]) -> Chroma:
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
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logger.info("Vector store created")
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

async def generate_stream_response(query: str, files: List[str]) -> AsyncGenerator[str, None]:
    logger.info(f"Generating response for query: {query}")
    try:
        if files:
            logger.info("Using RAG with files")
            vector_store = get_vector_store(files)
            
            # Create a streaming LLM
            llm = ChatOpenAI(
                temperature=0.7,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            
            # Create a prompt template
            template = """Answer the question based on the following context:
            {context}
            
            Question: {question}
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Create the RAG chain
            rag_chain = (
                {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Stream the response
            async for chunk in rag_chain.astream(query):
                print("chunk", chunk)
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
    logger.info(f"Received chat request - Prompt: {request.prompt}")
    logger.info(f"Files: {request.files}")
    return StreamingResponse(
        generate_stream_response(request.prompt, request.files),
        media_type="text/event-stream"
    )
    