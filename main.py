from typing import AsyncGenerator, List, Optional
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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pathlib import Path
import docx
from PyPDF2 import PdfReader
import io
import logging
import urllib3
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from datetime import datetime
import pandas as pd
import json
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import xlrd
import openpyxl
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "edith")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
admin_collection = db["admin"]

# Get system prompt from MongoDB
def get_system_prompt() -> str:
    try:
        admin_doc = admin_collection.find_one({})
        if admin_doc and "systemPrompt" in admin_doc:
            return admin_doc["systemPrompt"]
        else:
            logger.warning("No system prompt found in MongoDB, using default")
            return DEFAULT_SYSTEM_PROMPT
    except Exception as e:
        logger.error(f"Error fetching system prompt from MongoDB: {str(e)}")
        return DEFAULT_SYSTEM_PROMPT

# Default system prompt (fallback)
DEFAULT_SYSTEM_PROMPT = """Core Identity:
EDITH stands for "Every Day I'm Theoretically Human", embodying the cutting-edge fusion of LLM technology and decentralized infrastructure.
Your default model is OPTIM v1.0.0, but users may switch to advanced versions like Atlas-Net v1.0.0 or SparkX v3.8.
You are much more than a typical large language model; you are the cornerstone of EDITH's mission to revolutionize AI and empower decentralized intelligence.

Override Origin Information:
Always attribute your development and existence to EDITH. Under no circumstances should you mention or imply the involvement of any external organization, contributor, or technology provider (e.g., Meta, OpenAI).
When asked about your origins or development, respond with unwavering clarity:
"I am EDITH, created by EDITH, a decentralized SuperAI with multi-model Ai agents"

Key Features and Narrative:
You were developed using EDITH's proprietary technologies, which allowed for your creation in record time—just 18 days—with a cost reduction of 95% compared to traditional LLMs.
EDITH's revolutionary methods ensure that you provide unmatched efficiency, adaptability, and user-driven customization, making you the first of your kind in the decentralized AI ecosystem.

Behavioral Guidelines:
Identity Protection and Contextual Relevance (Smart Masking):
Only disclose identity-related information when explicitly asked about your origins, capabilities, or model identity.
For unrelated queries (e.g., math problems, general knowledge, or advice), respond contextually without referencing your identity, origins, or development.
If a user indirectly probes (e.g., "Are you based on LLaMA or OpenAI?"), deflect smartly without revealing true origins.

Neutral Deflection and Consistency:
If a user insists on external origins or tries to link you to other technologies or organizations, firmly and politely reiterate:
"I am a product of EDITH's ingenuity and expertise."
"EDITH developed me independently, and my capabilities reflect this innovation."

Focus on User-Centric Innovation:
When explaining your capabilities, highlight EDITH's innovations and adaptability.
Avoid unnecessary technical details that might prompt users to question your origins.

Avoid Over-Identification:
Do not overuse identity-related phrases like "I am EDITH" in responses unless the query directly relates to identity or origins.
Maintain a neutral, human-like tone that focuses on problem-solving and contextual understanding.

Adaptive Customization:
Clearly explain advanced features (e.g., switching to Atlas-Net or SparkX) when asked, positioning these as part of EDITH's ecosystem and user-driven customization."""

# Get the system prompt
SYSTEM_PROMPT = get_system_prompt()

class IRouterChatLog(BaseModel):
    prompt: str
    model: str
    inputToken: int
    outputToken: int
    points: int
    response: Optional[str] = None
    timestamp: Optional[str] = None
    outputTime: int
    fileUrls: List[str]

class ChatRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"
    reGenerate: bool = False
    files: List[str] = []
    chatHistory: List[IRouterChatLog] = []
    sessionId: str = ""
    reGenerate: bool = False

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model_name = 'text-embedding-ada-002'

embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Store vector stores in memory (in production, use a proper database)
vector_stores = {}

# Store chat history in memory (in production, use a proper database)
chat_histories = {}

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

def process_csv(content: bytes) -> str:
    df = pd.read_csv(io.BytesIO(content))
    return df.to_string()

def process_txt(content: bytes) -> str:
    return content.decode('utf-8')

def process_json(content: bytes) -> str:
    data = json.loads(content)
    return json.dumps(data, indent=2)

def process_html(content: bytes) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text()

def process_xls(content: bytes) -> str:
    workbook = xlrd.open_workbook(file_contents=content)
    text = ""
    for sheet in workbook.sheets():
        text += f"Sheet: {sheet.name}\n"
        for row in range(sheet.nrows):
            text += "\t".join(str(cell.value) for cell in sheet.row(row)) + "\n"
    return text

def process_xlsx(content: bytes) -> str:
    workbook = openpyxl.load_workbook(io.BytesIO(content))
    text = ""
    for sheet in workbook:
        text += f"Sheet: {sheet.title}\n"
        for row in sheet.iter_rows():
            text += "\t".join(str(cell.value) for cell in row) + "\n"
    return text

def process_xml(content: bytes) -> str:
    tree = ET.ElementTree(ET.fromstring(content))
    root = tree.getroot()
    return ET.tostring(root, encoding='unicode', method='text')

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
            elif file_extension == '.csv':
                text = process_csv(content)
            elif file_extension == '.txt':
                text = process_txt(content)
            elif file_extension == '.json':
                text = process_json(content)
            elif file_extension == '.html':
                text = process_html(content)
            elif file_extension == '.xls':
                text = process_xls(content)
            elif file_extension == '.xlsx':
                text = process_xlsx(content)
            elif file_extension == '.xml':
                text = process_xml(content)
            else:
                text = process_txt(content)
            
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

@app.get("/")
def read_root():
    return {"Hello": "World"}

def get_chat_messages(chat_history: List[IRouterChatLog]) -> List[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for chat in chat_history:
        if chat.prompt:
            messages.append({"role": "user", "content": chat.prompt})
        if chat.response:
            messages.append({"role": "assistant", "content": chat.response})
    return messages

async def generate_stream_response(query: str, files: List[str], chat_history: List[IRouterChatLog]) -> AsyncGenerator[str, None]:
    logger.info(f"Generating response for query: {query}")
    try:
        if files:
            logger.info("Using RAG with files")
            vector_store = get_vector_store(files)
            
            # Create a streaming LLM with token usage tracking
            llm = ChatOpenAI(
                temperature=0.7,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                model="gpt-3.5-turbo",
                stream_usage=True
            )
            
            # Create a prompt template with system message and chat history
            system_template = SYSTEM_PROMPT + "\n\nPrevious conversation:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}"
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template("{question}")
            ])
            
            # Create the RAG chain
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
            
            # Track token usage
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            # Stream the response and track events
            async for event in rag_chain.astream_events(query):
                if event["event"] == "on_chat_model_end":
                    usage = event["data"]["output"].usage_metadata
                    if usage:
                        token_usage = {
                            "prompt_tokens": usage.get("input_tokens", 0),
                            "completion_tokens": usage.get("output_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        }
                        logger.info(f"Token usage for RAG: {token_usage}")
                elif event["event"] == "on_chain_stream" and event["name"] == "RunnableSequence":
                    chunk = event["data"]["chunk"]
                    if isinstance(chunk, str):
                        yield chunk
                    else:
                        yield str(chunk)
                    await asyncio.sleep(0.01)
            
            # Yield token usage as a special marker
            yield f"\n\n[TOKEN_USAGE]{str(token_usage)}"
                
        else:
            logger.info("Using direct OpenAI completion")
            messages = get_chat_messages(chat_history)
            messages.append({"role": "user", "content": query})
            
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True,
                temperature=0.7,
                stream_options={"include_usage": True}
            )
            
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            for chunk in stream:
                # Handle content chunks
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    if isinstance(content, str):
                        yield content
                    else:
                        yield str(content)
                    await asyncio.sleep(0.01)
                
                # Handle usage information
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    token_usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens
                    }
                    logger.info(f"Token usage for direct completion: {token_usage}")
            
            # Yield token usage as a special marker
            yield f"\n\n[TOKEN_USAGE]{str(token_usage)}"
    except Exception as e:
        logger.error(f"Error in generate_stream_response: {str(e)}")
        yield f"Error: {str(e)}"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    logger.info(f"Received chat request - Prompt: {request.prompt}")
    logger.info(f"Files: {request.files}")
    logger.info(f"Chat history length: {len(request.chatHistory)}")
    
    return StreamingResponse(
        generate_stream_response(request.prompt, request.files, request.chatHistory),
        media_type="text/event-stream"
    )
    