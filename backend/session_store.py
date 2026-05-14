"""
Session persistence for anonymous browser and phone conversations.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from utils.config import get_config
from utils.logger import get_logger

logger = get_logger()
config = get_config()

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - optional dependency at runtime
    redis = None


class SessionStore:
    """Redis-backed session store with in-memory fallback."""

    def __init__(self):
        self._redis_client = None
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        self._backend = "memory"
        self._initialized = False

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def ttl_seconds(self) -> int:
        return config.session.ttl_seconds

    def _session_key(self, session_id: str) -> str:
        return f"{config.session.session_prefix}:session:{session_id}"

    def _session_index_key(self) -> str:
        return f"{config.session.session_prefix}:session_index"

    async def initialize(self):
        """Initialize Redis client if configured."""
        if self._initialized:
            return

        self._initialized = True
        if not config.session.is_redis_configured() or redis is None:
            if redis is None and config.session.is_redis_configured():
                logger.warning("Redis URL configured but redis package not installed. Using memory session store.")
            logger.info("Using in-memory session store")
            return

        try:
            self._redis_client = redis.from_url(config.session.redis_url, decode_responses=True)
            await self._redis_client.ping()
            self._backend = "redis"
            logger.info("Redis session store initialized")
        except Exception as error:
            logger.warning(f"Redis unavailable, falling back to memory store: {error}")
            self._redis_client = None
            self._backend = "memory"

    async def close(self):
        if self._redis_client is not None:
            await self._redis_client.close()
            self._redis_client = None

    async def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        await self.initialize()
        if self._backend == "redis" and self._redis_client is not None:
            payload = await self._redis_client.get(self._session_key(session_id))
            if not payload:
                return None
            try:
                data = json.loads(payload)
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                logger.warning(f"Corrupt session payload for {session_id}")
                return None

        payload = self._memory_store.get(session_id)
        if not payload:
            return None
        if time.time() - payload.get("updated_at", 0) > self.ttl_seconds:
            self._memory_store.pop(session_id, None)
            return None
        return payload

    async def save_session(self, session_id: str, data: Dict[str, Any]):
        await self.initialize()
        payload = dict(data)
        payload["session_id"] = session_id
        payload["updated_at"] = time.time()

        if self._backend == "redis" and self._redis_client is not None:
            serialized = json.dumps(payload, ensure_ascii=False)
            await self._redis_client.set(self._session_key(session_id), serialized, ex=self.ttl_seconds)
            await self._redis_client.zadd(self._session_index_key(), {session_id: payload["updated_at"]})
            await self._redis_client.zremrangebyscore(self._session_index_key(), 0, time.time() - self.ttl_seconds)
            return

        self._memory_store[session_id] = payload

    async def reset_session(self, session_id: str):
        await self.initialize()
        if self._backend == "redis" and self._redis_client is not None:
            await self._redis_client.delete(self._session_key(session_id))
            await self._redis_client.zrem(self._session_index_key(), session_id)
            return
        self._memory_store.pop(session_id, None)

    async def list_sessions(self, limit: int = 25) -> List[Dict[str, Any]]:
        await self.initialize()
        if self._backend == "redis" and self._redis_client is not None:
            session_ids = await self._redis_client.zrevrange(self._session_index_key(), 0, max(limit - 1, 0))
            sessions: List[Dict[str, Any]] = []
            for session_id in session_ids:
                session = await self.load_session(session_id)
                if session:
                    sessions.append(session)
            return sessions

        live_sessions = [
            session for session in self._memory_store.values()
            if time.time() - session.get("updated_at", 0) <= self.ttl_seconds
        ]
        live_sessions.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
        return live_sessions[:limit]

    async def get_active_count(self) -> int:
        sessions = await self.list_sessions(limit=500)
        return len(sessions)


_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
