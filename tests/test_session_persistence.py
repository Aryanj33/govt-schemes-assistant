import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


class FakeRAG:
    is_ready = True

    def build_index(self):
        return True

    async def search_parallel(self, query, top_k=5, filters=None, rerank=True):
        return [
            (
                {
                    "name": "Pragati Scholarship",
                    "benefits": "Up to Rs 50000 per year",
                    "eligibility": "SC students in engineering from Maharashtra colleges",
                    "application_process": "Apply through the official AICTE portal",
                    "documents": ["income certificate", "marksheet"],
                },
                0.91,
            )
        ]


@pytest.fixture
def conversation_module(monkeypatch):
    import backend.agent.conversation_handler as ch

    monkeypatch.setattr(ch, "get_scholarship_rag", lambda: FakeRAG())
    monkeypatch.setattr(ch.config.groq, "is_configured", lambda: False)
    monkeypatch.setattr(ch.config.google, "is_gemini_configured", lambda: False)
    return ch


@pytest.mark.asyncio
async def test_session_state_persists_when_handler_is_recreated(conversation_module):
    from backend.session_store import get_session_store

    session_id = "persist-test-session"
    store = get_session_store()
    await store.reset_session(session_id)
    conversation_module._conversation_handlers.pop(session_id, None)
    conversation_module._handler_timestamps.pop(session_id, None)

    handler = conversation_module.get_conversation_handler(session_id)
    await handler.generate_response("I need scholarship information", language="english")
    await handler.generate_response("I am from Maharashtra and SC category", language="english")
    await handler.generate_response("I study engineering", language="english")

    conversation_module._conversation_handlers.pop(session_id, None)
    conversation_module._handler_timestamps.pop(session_id, None)

    restored_handler = conversation_module.get_conversation_handler(session_id)
    await restored_handler.initialize()

    assert restored_handler.state.profile.state == "Maharashtra"
    assert restored_handler.state.profile.category == "SC"
    assert restored_handler.state.profile.course == "Engineering"
    assert restored_handler.state.turn_count >= 3
    assert restored_handler.state.messages
