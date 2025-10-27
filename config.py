import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Gemini AI
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Database (NeonDB)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # Redis (Upstash)
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    
    # Application
    PROD_URL: str = os.getenv("PROD_URL", "http://localhost:8000")

settings = Settings()

