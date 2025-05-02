# FastAPI Chatbot with RAG

A modern chatbot application built with FastAPI that supports Retrieval-Augmented Generation (RAG) and file processing capabilities.

## Features

- Real-time streaming chat responses
- Support for multiple file formats (PDF, DOCX, CSV, TXT, JSON, HTML, XLS, XLSX, XML)
- RAG (Retrieval-Augmented Generation) for enhanced responses
- MongoDB integration for chat history and configuration
- Token usage tracking and cost calculation
- CORS support for cross-origin requests

## Project Structure

```
app/
├── api/            # API routes and endpoints
├── config/         # Configuration settings
├── core/           # Core functionality and database
├── models/         # Pydantic models
├── services/       # Business logic and services
└── utils/          # Utility functions and helpers
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your configuration:
   ```
   MONGODB_URI=mongodb://localhost:27017
   DB_NAME=edith
   OPENAI_API_KEY=your_openai_api_key
   ```

## Running the Application

Start the server with:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### POST /api/chat/stream
Stream chat responses with support for file processing and RAG.

Request body:
```json
{
  "prompt": "Your question here",
  "model": "gpt-3.5-turbo",
  "reGenerate": false,
  "files": ["http://example.com/file.pdf"],
  "chatHistory": [],
  "sessionId": "unique-session-id",
  "email": "user@example.com"
}
```

## Development

- The project uses FastAPI for the web framework
- MongoDB for data storage
- OpenAI's API for chat completions
- LangChain for RAG implementation
- Pydantic for data validation

## License

MIT 