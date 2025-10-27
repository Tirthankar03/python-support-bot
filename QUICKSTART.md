# Quick Start Guide

## 🚀 Get Your IT Support Chatbot Running in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Up Environment Variables
```bash
# Copy the example file
cp env.example .env

# Edit .env with your credentials
# You'll need:
# - GEMINI_API_KEY (from Google AI Studio)
# - DATABASE_URL (from NeonDB)
# - REDIS_URL (from Upstash)
```

### Step 3: Set Up Database
```bash
python setup_database.py
```

### Step 4: Run the Application
```bash
python run.py
```

### Step 5: Test the API
```bash
# In another terminal
python test_api.py
```

## 🧪 Test with curl

```bash
# Test the main support endpoint
curl -X POST "http://localhost:8000/support" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "My laptop won't turn on",
    "chatId": "test_user_123",
    "name": "Test User",
    "history": []
  }'
```

## 📊 Check API Documentation

Visit: http://localhost:8000/docs

## 🔧 Troubleshooting

1. **Database Connection Issues**: Check your NeonDB connection string
2. **Redis Connection Issues**: Verify your Upstash Redis URL
3. **AI Not Responding**: Check your Gemini API key
4. **Import Errors**: Make sure all dependencies are installed

## 📝 What You Get

- ✅ AI-powered IT support responses
- ✅ Automatic ticket creation
- ✅ Conversation history with Redis
- ✅ Safety guardrails with NVIDIA Guardrails
- ✅ Clean REST API endpoints
- ✅ PostgreSQL database for data persistence

That's it! Your IT support chatbot is ready to use. 🎉
