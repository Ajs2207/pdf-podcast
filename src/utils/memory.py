import json
import redis
from config.settings import REDIS_URL


class RedisChatMemory:
    def __init__(self):
        self.client = redis.from_url(REDIS_URL, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"chat:{session_id}"

    def get_history(self, session_id: str):
        data = self.client.get(self._key(session_id))
        return json.loads(data) if data else []

    def append(self, session_id: str, role: str, content: str):
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        self.client.set(self._key(session_id), json.dumps(history))
