"""
End-to-End Validation Suite
===========================

Exercises every critical path of the assistant in a single, deterministic
test run. Each section targets one capability of the system:

  1. Configuration loading and voice-vs-text model split.
  2. Scholarship RAG bootstrap and dedup pipeline (regression for the
     ``re`` scoping bug in ``_get_content_hash``).
  3. Profile extraction (state, category, course, name, gender, marks
     regression for the ``%`` word-boundary bug, income parsing).
  4. Conversation flow:
       a. multi-turn profile gathering produces the right follow-up
          questions in order.
       b. a single multi-fact utterance short-circuits straight to
          grounded results.
       c. specific scheme lookups (PM-KISAN, Mudra, Pragati) skip the
          profile question and answer immediately.
       d. detail follow-ups (eligibility / documents / apply) are served
          from the cached scheme list.
  5. Search endpoint returns both the new and legacy result shapes and
     applies category alias + strict state filtering.
  6. Voice streaming pipeline:
       a. uses the voice model + token cap from config.
       b. on ``finish_reason='length'`` the trailing partial is dropped
          and a graceful closer is emitted instead.
       c. on a clean stop short trailing fragments (e.g. ``"Up to"``)
          are discarded so TTS never speaks a clipped tail.
       d. soft-break flushing on commas/colons fires before the first
          full stop when the clause is long enough.
  7. Session persistence survives handler eviction (including writes
     that are still in-flight via the ``_pending_persists`` table).
  8. WebSocket message contract for the voice endpoint shape.

The suite uses small in-memory fakes for the RAG index, the Groq client,
the session store, and the websocket so it runs in seconds without
touching any external service or API key.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import pytest

# Make ``backend`` importable as a top-level package the same way the rest
# of the test suite does.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# --------------------------------------------------------------------------- #
# Lightweight fakes                                                           #
# --------------------------------------------------------------------------- #

PRAGATI = {
    "id": "central-pragati",
    "name": "Pragati Scholarship",
    "details": "AICTE Pragati for girl students in technical degree courses.",
    "benefits": "Up to Rs 50000 per year for tuition, books, and incidentals.",
    "eligibility": "SC/ST/OBC girl students enrolled in approved engineering colleges.",
    "application_process": "Apply through the official AICTE Pragati portal.",
    "documents": ["income certificate", "marksheet", "Aadhaar"],
    "level": "Central",
    "tags": ["Scholarship", "Engineering", "Girls"],
    "category": ["Scholarship", "Women"],
}

PM_KISAN = {
    "id": "central-pm-kisan",
    "name": "Pradhan Mantri Kisan Samman Nidhi",
    "details": "Income support to all eligible landholding farmer families.",
    "benefits": "Rs 6000 per year direct benefit transfer in three instalments.",
    "eligibility": "Landholding farmer families across India.",
    "application_process": "Apply through the PM Kisan portal or CSC.",
    "documents": ["land record", "Aadhaar", "bank account"],
    "level": "Central",
    "tags": ["Kisan", "Farmer"],
}


class FakeRAG:
    """In-memory RAG that supports both alias lookups and parallel search."""

    is_ready = True
    scholarships = [PRAGATI, PM_KISAN]
    vectorstore = SimpleNamespace(documents=[PRAGATI, PM_KISAN])

    def build_index(self) -> bool:  # pragma: no cover - trivial
        return True

    async def search_parallel(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ) -> List[Tuple[Dict[str, Any], float]]:
        lowered = query.lower()
        if "pm-kisan" in lowered or "pm kisan" in lowered or "kisan samman" in lowered:
            return [(PM_KISAN, 0.96)]
        if "kisan" in lowered or "farmer" in lowered:
            return [(PM_KISAN, 0.85)]
        return [(PRAGATI, 0.91)]


class FakeStreamChoice:
    def __init__(self, content: Optional[str], finish_reason: Optional[str] = None):
        self.delta = SimpleNamespace(content=content)
        self.finish_reason = finish_reason


class FakeStreamChunk:
    def __init__(self, content: Optional[str], finish_reason: Optional[str] = None):
        self.choices = [FakeStreamChoice(content, finish_reason)]


class FakeStream:
    """Async iterator that yields the configured chunks."""

    def __init__(self, chunks: List[FakeStreamChunk]):
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[FakeStreamChunk]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[FakeStreamChunk]:
        for chunk in self._chunks:
            yield chunk


class FakeChatCompletions:
    """Records every call and returns the queued stream/message."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.next_stream: Optional[List[FakeStreamChunk]] = None

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            chunks = self.next_stream or []
            self.next_stream = None
            return FakeStream(chunks)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )


class FakeGroqClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def conversation_module(monkeypatch):
    """Wire the conversation module up to fakes only."""
    import backend.agent.conversation_handler as ch

    monkeypatch.setattr(ch, "get_scholarship_rag", lambda: FakeRAG())
    monkeypatch.setattr(ch.config.groq, "is_configured", lambda: False)
    monkeypatch.setattr(ch.config.google, "is_gemini_configured", lambda: False)
    return ch


@pytest.fixture
def streaming_module(monkeypatch):
    """Variant that still uses fakes but installs a fake Groq client."""
    import backend.agent.conversation_handler as ch

    monkeypatch.setattr(ch, "get_scholarship_rag", lambda: FakeRAG())
    monkeypatch.setattr(ch.config.groq, "is_configured", lambda: True)
    monkeypatch.setattr(ch.config.google, "is_gemini_configured", lambda: False)
    return ch


# --------------------------------------------------------------------------- #
# 1. Configuration                                                            #
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_voice_and_text_models_are_distinct(self):
        from backend.utils.config import get_config

        cfg = get_config()
        # Text path stays on the bigger 70B model; voice uses 8b-instant.
        assert "70b" in cfg.groq.llm_model.lower()
        assert "8b" in cfg.groq.voice_llm_model.lower()
        # Voice cap is large enough to avoid mid-sentence cut-offs but still
        # below the text cap.
        assert cfg.groq.voice_llm_max_tokens >= 200
        assert cfg.groq.voice_llm_max_tokens <= cfg.groq.llm_max_tokens + 50

    def test_cartesia_uses_multilingual_sonic_2(self):
        from backend.utils.config import get_config

        assert get_config().cartesia.model_id.startswith("sonic-2")


# --------------------------------------------------------------------------- #
# 2. RAG bootstrap (regression for the ``re`` scoping bug)                    #
# --------------------------------------------------------------------------- #


class TestRAGBootstrap:
    def test_get_content_hash_uses_module_level_re(self):
        """Regression: previously raised NameError on ``re``."""
        from backend.rag.scholarship_rag import get_scholarship_rag

        rag = get_scholarship_rag()
        # _get_content_hash is private but stable enough to validate.
        h1 = rag._get_content_hash(PRAGATI)
        h2 = rag._get_content_hash(PRAGATI)
        h3 = rag._get_content_hash(PM_KISAN)
        assert isinstance(h1, str) and len(h1) == 32
        assert h1 == h2  # deterministic
        assert h1 != h3  # content-sensitive

    def test_dedup_keeps_distinct_schemes(self):
        from backend.rag.scholarship_rag import get_scholarship_rag

        rag = get_scholarship_rag()
        deduped = rag._deduplicate_scholarships(
            [PRAGATI, PRAGATI.copy(), PM_KISAN]
        )
        names = {scheme["name"] for scheme in deduped}
        assert names == {PRAGATI["name"], PM_KISAN["name"]}


# --------------------------------------------------------------------------- #
# 3. Profile extraction                                                        #
# --------------------------------------------------------------------------- #


class TestProfileExtraction:
    def test_state_category_marks_in_one_message(self, conversation_module):
        """Regression for the marks-percentage regex bug."""
        ch = conversation_module
        profile = ch.UserProfile()
        ch.extract_profile_from_message(
            profile, "I'm from Maharashtra, SC category, 85% marks"
        )
        assert profile.state == "Maharashtra"
        assert profile.category == "SC"
        assert profile.marks == 85.0

    @pytest.mark.parametrize(
        "phrase,expected",
        [
            ("scored 92% in boards", 92.0),
            ("got 78 percent", 78.0),
            ("85 percentage marks", 85.0),
            ("just 45%.", 45.0),
        ],
    )
    def test_marks_recognised_in_various_phrasings(
        self, conversation_module, phrase, expected
    ):
        ch = conversation_module
        profile = ch.UserProfile()
        ch.extract_profile_from_message(profile, phrase)
        assert profile.marks == expected

    def test_name_extraction_and_rejection_of_phrases(self, conversation_module):
        ch = conversation_module
        positive = ch.UserProfile()
        ch.extract_profile_from_message(positive, "my name is Aryan")
        assert positive.name and positive.name.lower() == "aryan"

        # Should NOT capture "mission" / "name" / state names as user names.
        rejects = [
            "mera mission scholarship hai",
            "i am from karnataka",
            "i am studying engineering",
        ]
        for utterance in rejects:
            profile = ch.UserProfile()
            ch.extract_profile_from_message(profile, utterance)
            assert profile.name is None, f"unexpected name for {utterance!r}: {profile.name}"

    def test_income_parses_lakh_and_thousand(self, conversation_module):
        ch = conversation_module
        for utterance, expected in [
            ("family income 1.5 lakh", 150000),
            ("annual income 80 thousand", 80000),
            ("income 120000", 120000),
        ]:
            profile = ch.UserProfile()
            ch.extract_profile_from_message(profile, utterance)
            assert profile.income == expected

    def test_course_and_gender_capture(self, conversation_module):
        ch = conversation_module
        profile = ch.UserProfile()
        ch.extract_profile_from_message(profile, "BTech engineering girl student")
        assert profile.course == "Engineering"
        assert profile.gender == "Female"


# --------------------------------------------------------------------------- #
# 4. Conversation flow                                                        #
# --------------------------------------------------------------------------- #


class TestConversationFlow:
    @pytest.mark.asyncio
    async def test_multi_turn_profile_gathering_then_results(self, conversation_module):
        handler = conversation_module.ConversationHandler()
        r1 = await handler.generate_response("I need scholarship information", language="english")
        r2 = await handler.generate_response("I am from Maharashtra", language="english")
        r3 = await handler.generate_response("SC category", language="english")
        r4 = await handler.generate_response("I study engineering", language="english")

        assert "state" in r1.lower()
        assert "category" in r2.lower()
        assert "course" in r3.lower()
        assert "Pragati Scholarship" in r4

    @pytest.mark.asyncio
    async def test_single_utterance_with_full_profile_short_circuits_to_results(
        self, conversation_module
    ):
        handler = conversation_module.ConversationHandler()
        response = await handler.generate_response(
            "Maharashtra SC engineering scholarship",
            language="english",
        )
        # Profile complete in one breath -> grounded results immediately,
        # no "which state are you from?" follow-up.
        assert "Pragati Scholarship" in response
        assert "which state" not in response.lower()

    @pytest.mark.asyncio
    async def test_specific_scheme_lookup_skips_profile_questions(self, conversation_module):
        handler = conversation_module.ConversationHandler()
        response = await handler.generate_response(
            "PM-KISAN me kitna paisa milta hai",
            language="hinglish",
        )
        assert "Pradhan Mantri Kisan Samman Nidhi" in response
        assert "6000" in response

    @pytest.mark.asyncio
    async def test_detail_follow_up_returns_eligibility_from_cached_results(
        self, conversation_module
    ):
        handler = conversation_module.ConversationHandler()
        # Prime the cache with results.
        await handler.generate_response(
            "Maharashtra SC engineering scholarship", language="english"
        )
        # Follow-up keyed on "eligibility".
        response = await handler.generate_response(
            "what is the eligibility for the first one",
            language="english",
        )
        assert "eligibil" in response.lower()
        assert "Pragati" in response


# --------------------------------------------------------------------------- #
# 5. Search endpoint shape and filters                                        #
# --------------------------------------------------------------------------- #


class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search_returns_legacy_and_new_shapes(self, monkeypatch):
        import backend.server as server

        class Handler:
            def __init__(self):
                self.rag = FakeRAG()

            async def initialize(self):
                return None

            state = SimpleNamespace(get_profile_summary=lambda: "No profile info yet")

        monkeypatch.setattr(server, "get_conversation_handler", lambda session_id="default": Handler())

        result = await server.search_schemes(
            server.SearchRequest(query="engineering scholarship")
        )
        assert result["schemes"]
        assert result["results"]
        # Same scheme in both shapes
        assert result["results"][0][0]["name"] == result["schemes"][0]["name"]
        # Legacy fields are populated
        scheme = result["schemes"][0]
        assert scheme["description"] == scheme["details"]
        assert scheme["amount"] == scheme["benefits"]


# --------------------------------------------------------------------------- #
# 6. Voice streaming pipeline                                                 #
# --------------------------------------------------------------------------- #


class TestVoiceStreaming:
    @pytest.mark.asyncio
    async def test_streaming_uses_voice_model_and_token_cap(self, streaming_module):
        ch = streaming_module
        handler = ch.ConversationHandler()
        handler.groq_client = FakeGroqClient()
        # Force the LLM-streaming path by skipping the deterministic builders.
        handler._build_direct_response = lambda *args, **kwargs: None  # type: ignore[assignment]

        handler.groq_client.chat.completions.next_stream = [
            FakeStreamChunk("Hello there.", finish_reason="stop"),
        ]

        chunks: List[str] = []
        async for piece in handler.generate_response_stream("any question"):
            chunks.append(piece)

        call = handler.groq_client.chat.completions.calls[-1]
        assert call["model"] == ch.config.groq.voice_llm_model
        assert call["max_tokens"] == ch.config.groq.voice_llm_max_tokens
        assert call["stream"] is True
        assert chunks == ["Hello there."]

    @pytest.mark.asyncio
    async def test_finish_reason_length_appends_graceful_closer(self, streaming_module):
        """Regression: AI used to stop mid-word when max_tokens hit."""
        ch = streaming_module
        handler = ch.ConversationHandler()
        handler.groq_client = FakeGroqClient()
        handler._build_direct_response = lambda *args, **kwargs: None  # type: ignore[assignment]
        handler.state.profile.language = "hinglish"

        # The model produces one full sentence then a partial mid-word.
        handler.groq_client.chat.completions.next_stream = [
            FakeStreamChunk("Pragati Scholarship is for engineering students."),
            FakeStreamChunk(" Benefit"),
            FakeStreamChunk(" is up to", finish_reason="length"),
        ]

        chunks: List[str] = []
        async for piece in handler.generate_response_stream("tell me about pragati"):
            chunks.append(piece)

        # The full sentence is yielded, but the partial trailing buffer is
        # NOT yielded as-is. Instead a graceful closer takes its place.
        joined = " ".join(chunks)
        assert "Pragati Scholarship is for engineering students" in joined
        assert "is up to" not in joined  # partial fragment dropped
        assert "puri detail" in joined or "full details" in joined  # closer added

    @pytest.mark.asyncio
    async def test_clean_stop_drops_short_unpunctuated_tail(self, streaming_module):
        """Stray fragments like 'Up to' should not be spoken at the end."""
        ch = streaming_module
        handler = ch.ConversationHandler()
        handler.groq_client = FakeGroqClient()
        handler._build_direct_response = lambda *args, **kwargs: None  # type: ignore[assignment]

        handler.groq_client.chat.completions.next_stream = [
            FakeStreamChunk("Aap ke liye scholarship available hai."),
            FakeStreamChunk(" Up to", finish_reason="stop"),
        ]

        chunks: List[str] = []
        async for piece in handler.generate_response_stream("scholarship?"):
            chunks.append(piece)

        joined = " ".join(chunks)
        assert "scholarship available hai" in joined
        # The clipped " Up to" tail is short, lacks punctuation, and is
        # discarded by the trailing-buffer guard.
        assert chunks[-1].endswith(".") or chunks[-1].endswith("hai.")

    @pytest.mark.asyncio
    async def test_soft_break_flushes_long_clause_before_full_stop(self, streaming_module):
        """A long clause ending in a comma should flush early for low-latency TTS."""
        ch = streaming_module
        handler = ch.ConversationHandler()
        handler.groq_client = FakeGroqClient()
        handler._build_direct_response = lambda *args, **kwargs: None  # type: ignore[assignment]

        # The first clause is >28 chars and ends in a comma -> should flush
        # before the full stop arrives.
        handler.groq_client.chat.completions.next_stream = [
            FakeStreamChunk("Aap ke liye Pragati Scholarship available hai,"),
            FakeStreamChunk(" jiska benefit Rs 50000 tak hai.", finish_reason="stop"),
        ]

        chunks: List[str] = []
        async for piece in handler.generate_response_stream("scholarship?"):
            chunks.append(piece)

        # We should see at least 2 yielded sentences thanks to the soft break.
        assert len(chunks) >= 2
        assert chunks[0].endswith(",")


# --------------------------------------------------------------------------- #
# 7. Session persistence + pending-persist tracking                           #
# --------------------------------------------------------------------------- #


class TestSessionPersistence:
    @pytest.mark.asyncio
    async def test_state_survives_handler_eviction(self, conversation_module):
        ch = conversation_module
        from backend.session_store import get_session_store

        session_id = "e2e-validation-session"
        store = get_session_store()
        await store.reset_session(session_id)
        ch._conversation_handlers.pop(session_id, None)
        ch._handler_timestamps.pop(session_id, None)
        ch._pending_persists.pop(session_id, None)

        h1 = ch.get_conversation_handler(session_id)
        await h1.generate_response("Maharashtra SC engineering scholarship", language="english")

        # Evict the in-memory handler; pending persist may still be running.
        ch._conversation_handlers.pop(session_id, None)
        ch._handler_timestamps.pop(session_id, None)

        h2 = ch.get_conversation_handler(session_id)
        await h2.initialize()

        assert h2.state.profile.state == "Maharashtra"
        assert h2.state.profile.category == "SC"
        assert h2.state.profile.course == "Engineering"
        assert h2.state.profile.scheme_type == "scholarship"
        assert h2.state.last_scholarships  # results were preserved


# --------------------------------------------------------------------------- #
# 8. WebSocket voice endpoint contract                                        #
# --------------------------------------------------------------------------- #


class TestWebSocketContract:
    """The streaming pipeline talks to the client over a small JSON +
    binary protocol. We capture every message the server would send for a
    single audio turn and assert the contract."""

    @pytest.mark.asyncio
    async def test_streaming_pipeline_sends_expected_messages(self, monkeypatch, streaming_module):
        import backend.server as server

        # Wire the handler used by the websocket helper to our fakes.
        ch = streaming_module
        handler = ch.ConversationHandler(session_id="ws-test")
        handler.groq_client = FakeGroqClient()
        handler.groq_client.chat.completions.next_stream = [
            FakeStreamChunk("Pragati Scholarship for engineering girls.", finish_reason="stop"),
        ]
        handler._build_direct_response = lambda *args, **kwargs: None  # type: ignore[assignment]

        # Skip real STT entirely - patch the inner streaming helper.
        async def fake_tts(sentence, lang="hindi"):
            return b"AUDIO:" + sentence.encode("utf-8")

        monkeypatch.setattr(server, "_generate_tts_for_sentence_cached", fake_tts)

        # Whisper audio.transcriptions.create -> return a canned transcript.
        async def fake_transcribe(**kwargs):
            return "engineering scholarship"

        handler.groq_client.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=fake_transcribe)
        )

        sent_text: List[str] = []
        sent_bytes: List[bytes] = []

        class FakeWebSocket:
            async def send_text(self, payload: str):
                sent_text.append(payload)

            async def send_bytes(self, payload: bytes):
                sent_bytes.append(payload)

        # Make sure the streaming pipeline picks up our scripted handler.
        monkeypatch.setattr(server, "get_conversation_handler", lambda sid="default": handler)

        # Build the message and call the streaming entrypoint directly.
        import base64

        message = {"audio_data": base64.b64encode(b"\x1a\x45\xdf\xa3" + b"x" * 200).decode("utf-8")}
        await server.process_complete_audio_streaming(
            FakeWebSocket(), message, handler, "ws-test"
        )

        # We expect: at least one transcription frame, one response frame,
        # one audio_complete frame, and one binary chunk.
        kinds = [json.loads(t).get("type") for t in sent_text]
        assert "transcription" in kinds
        assert "response" in kinds
        assert "audio_complete" in kinds
        assert sent_bytes  # binary audio was sent
        assert sent_bytes[0].startswith(b"AUDIO:")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "--tb=short"])
