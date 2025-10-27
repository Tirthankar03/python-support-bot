#!/usr/bin/env python3
"""
Startup script for the IT Support Chatbot
"""
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Check if required environment variables are set
    required_vars = ["GEMINI_API_KEY", "DATABASE_URL", "REDIS_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        print("See env.example for reference.")
        exit(1)
    
    print("🚀 Starting IT Support Chatbot...")
    print("📱 API will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("🔧 Health check: http://localhost:8000/health")
    print()
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
