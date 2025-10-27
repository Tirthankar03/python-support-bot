import redis
import json
from typing import List, Dict, Any
from datetime import datetime
from config import settings

class RedisClient:
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
    
    async def save_message(self, chat_id: str, message: str, is_bot: bool) -> None:
        """Save a message to Redis conversation history"""
        msg = {
            "message": message,
            "isBot": is_bot,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        self.redis_client.lpush(f"user:{chat_id}:history", json.dumps(msg))
        self.redis_client.ltrim(f"user:{chat_id}:history", 0, 99)  # Keep last 100 messages
    
    async def get_conversation_history(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        messages = self.redis_client.lrange(f"user:{chat_id}:history", 0, 9)  # Get last 10
        return [json.loads(msg) for msg in reversed(messages)]  # Reverse to chronological order
    
    async def clear_history(self, chat_id: str) -> None:
        """Clear conversation history for a user"""
        self.redis_client.delete(f"user:{chat_id}:history")

# Global Redis client instance
redis_client = RedisClient()

