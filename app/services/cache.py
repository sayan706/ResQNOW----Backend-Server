import os
import json
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class RedisCache:
    def __init__(self):
        self.client = redis.from_url(REDIS_URL, decode_responses=True)

    def get(self, key):
        data = self.client.get(key)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        return None

    def set(self, key, value, expire=3600):
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        self.client.set(key, value, ex=expire)

    def delete(self, key):
        self.client.delete(key)

cache_service = RedisCache()
