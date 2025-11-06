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

    # WhatsApp Business API
    ACCESS_TOKEN: str = os.getenv("ACCESS_TOKEN", "")
    APP_ID: str = os.getenv("APP_ID", "")
    APP_SECRET: str = os.getenv("APP_SECRET", "")
    RECIPIENT_WAID: str = os.getenv("RECIPIENT_WAID", "")
    VERSION: str = os.getenv("VERSION", "v18.0")
    PHONE_NUMBER_ID: str = os.getenv("PHONE_NUMBER_ID", "")
    VERIFY_TOKEN: str = os.getenv("VERIFY_TOKEN", "")

    # Application
    PROD_URL: str = os.getenv("PROD_URL", "http://localhost:8000")

settings = Settings()

