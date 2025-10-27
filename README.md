# IT Support Chatbot

A simple AI-powered IT support chatbot built with FastAPI, featuring long-term memory, ticket management, and a lightweight internal guardrails implementation for safety (no external guardrails library required).

## Features

- **AI-Powered Support**: Uses Google Gemini 1.5 Flash for intelligent IT support responses
- **Long-term Memory**: Redis-based conversation history for personalized interactions
- **Ticket Management**: Automatic ticket creation and tracking via PostgreSQL (NeonDB)
-- **Safety Guardrails**: Lightweight internal guardrails for content safety and IT-specific filtering (no external library required)
- **Function Calling**: AI can create tickets, retrieve past tickets, and get current date
- **REST API**: Clean FastAPI endpoints for easy integration

## Tech Stack

- **Backend**: FastAPI (Python)
- **AI Model**: Google Gemini 1.5 Flash
- **Database**: PostgreSQL (NeonDB)
- **Cache**: Redis (Upstash)
-- **Safety**: Lightweight internal guardrails (no external library required)

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Copy `env.example` to `.env` and fill in your credentials:
   ```bash
   cp env.example .env
   ```

   Required environment variables:
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `DATABASE_URL`: PostgreSQL connection string (NeonDB)
   - `REDIS_URL`: Redis connection string (Upstash)
   - `PROD_URL`: Your production URL (optional, defaults to localhost:8000)

3. **Database Setup**
   Run the database setup script to create the required tables:
   ```bash
   python setup_database.py
   ```

4. **Run the Application**
   ```bash
   python run.py
   ```

   Or with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints

### POST `/support`
Main endpoint for processing IT support queries.

**Request Body:**
```json
{
  "query": "My laptop won't turn on",
  "chatId": "user123",
  "name": "John Doe",
}
```

**Response:**
```json
{
  "response": "I'll help you troubleshoot your laptop issue. Let me create a ticket for this..."
}
```

### GET `/tickets/{chat_id}`
Get all tickets for a specific user.

### GET `/history/{chat_id}`
Get conversation history for a user.

### DELETE `/history/{chat_id}`
Clear conversation history for a user.

## Making API Requests

You can test the API endpoints using curl commands:

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Submit Support Query
```bash
curl -X POST http://localhost:8000/support \
  -H "Content-Type: application/json" \
  -d '{
    "query": "My laptop won'\''t turn on",
    "chatId": "user123",
    "name": "John Doe",
  }'
```

### 3. Get User's Tickets
```bash
curl http://localhost:8000/tickets/user123
```

### 4. Get Conversation History
```bash
curl http://localhost:8000/history/user123
```

### 5. Clear Conversation History
```bash
curl -X DELETE http://localhost:8000/history/user123
```

Note: Replace `user123` with your actual chatId when making requests.

## Testing

Use the included test script to verify the API works correctly:

```bash
python test_api.py
```

This will test all endpoints and verify the AI responses are working properly.

## Safety Features

- **Internal Guardrails**: Lightweight checks for content safety and filtering (designed for assignment use; not a replacement for production guardrail libraries)
- **IT-Specific Filtering**: Blocks non-IT related queries
- **Sensitive Data Protection**: Prevents processing of passwords, credit cards, etc.
- **Harmful Action Prevention**: Blocks suggestions for unsafe actions

## Database Schema

### Users Table
- `user_id` (String, Primary Key)
- `name` (String)
- `created_at` (DateTime)
- `updated_at` (DateTime)

### Tickets Table
- `ticket_id` (Integer, Primary Key, Auto-increment)
- `user_id` (String, Foreign Key)
- `problem_description` (Text)
- `solution_suggested` (Text)
- `status` (String, Default: "open")
- `created_at` (DateTime)
- `updated_at` (DateTime)

## Development

The application is structured as follows:
- `main.py`: FastAPI application and endpoints
- `ai_agent.py`: AI agent with function calling capabilities
- `database.py`: Database models and connection
- `redis_client.py`: Redis client for conversation history
 - `guardrails_config.py`: Lightweight internal guardrails configuration
- `config.py`: Application configuration
- `setup_database.py`: Database schema setup script
- `run.py`: Application startup script
- `test_api.py`: API testing script

## License

MIT License
