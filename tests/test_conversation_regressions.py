import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


class FakeRAG:
    is_ready = True
    scholarships = [
        {
            "name": "Pradhan Mantri Kisan Samman Nidhi",
            "benefits": "Rs 6000 per year",
            "eligibility": "All eligible landholding farmers",
            "application_process": "Apply through PM Kisan portal",
            "level": "Central",
            "tags": ["Kisan", "Farmer"],
        },
        {
            "name": "Pragati Scholarship",
            "benefits": "Up to Rs 50000 per year",
            "eligibility": "SC/ST/OBC girl students in engineering from approved colleges",
            "application_process": "Apply through the official AICTE portal",
            "documents": ["income certificate", "marksheet"],
            "level": "Central",
            "tags": ["Scholarship", "Engineering"],
        },
    ]

    def build_index(self):
        return True

    async def search_parallel(self, query, top_k=5, filters=None, rerank=True):
        if "pm-kisan" in query.lower() or "pm kisan" in query.lower():
            return [
                (
                    {
                        "name": "Mukhyamantri Kisaan Kalyaan Yojana",
                        "benefits": "Rs 4000 per year",
                        "eligibility": "Madhya Pradesh farmers registered under PM Kisan",
                        "application_process": "Apply through state agriculture office",
                        "level": "State",
                    },
                    0.92,
                )
            ]

        return [
            (
                {
                    "name": "Pragati Scholarship",
                    "benefits": "Up to Rs 50000 per year",
                    "eligibility": "SC/ST/OBC girl students in engineering from approved colleges",
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
async def test_profile_progression_stays_in_one_session(conversation_module):
    handler = conversation_module.ConversationHandler()

    response_1 = await handler.generate_response("I need scholarship information", language="english")
    response_2 = await handler.generate_response("I am from Maharashtra", language="english")
    response_3 = await handler.generate_response("SC category", language="english")
    response_4 = await handler.generate_response("I study engineering", language="english")

    assert "state" in response_1.lower()
    assert "category" in response_2.lower()
    assert "course" in response_3.lower()
    assert "Pragati Scholarship" in response_4
    assert handler.state.profile.state == "Maharashtra"
    assert handler.state.profile.category == "SC"
    assert handler.state.profile.course == "Engineering"


@pytest.mark.asyncio
async def test_specific_scheme_lookup_does_not_wait_for_full_profile(conversation_module):
    handler = conversation_module.ConversationHandler()

    response = await handler.generate_response("PM-KISAN me kitna paisa milta hai", language="hinglish")

    assert "Pradhan Mantri Kisan Samman Nidhi" in response
    assert "6000" in response


@pytest.mark.asyncio
async def test_search_endpoint_keeps_legacy_results_shape(monkeypatch):
    import backend.server as server

    class FakeHandler:
        def __init__(self):
            self.rag = FakeRAG()

        async def initialize(self):
            return None

    monkeypatch.setattr(server, "get_conversation_handler", lambda session_id="default": FakeHandler())

    response = await server.search_schemes(server.SearchRequest(query="engineering scholarship"))

    assert response["schemes"]
    assert response["results"]
    assert response["results"][0][0]["name"] == response["schemes"][0]["name"]
    assert response["schemes"][0]["description"] == response["schemes"][0]["details"]


@pytest.mark.asyncio
async def test_search_endpoint_supports_category_alias_and_strict_state_filter(monkeypatch):
    import backend.server as server

    class FakeHandler:
        def __init__(self):
            self.rag = FakeRAG()

        async def initialize(self):
            return None

    async def fake_search_parallel(query, top_k=5, filters=None, rerank=True):
        return [
            (
                {
                    "id": "central-sc",
                    "name": "National SC Scholarship",
                    "details": "Central scholarship for Scheduled Caste students across India.",
                    "benefits": "Rs 10000",
                    "eligibility": "Scheduled Caste students",
                    "level": "Central",
                    "category": ["Scholarship", "Scheduled Caste"],
                    "tags": ["SC", "Student"],
                },
                0.91,
            ),
            (
                {
                    "id": "mh-sc",
                    "name": "Maharashtra SC Engineering Scholarship",
                    "details": "Scholarship for Scheduled Caste engineering students in Maharashtra.",
                    "benefits": "Rs 50000",
                    "eligibility": "Scheduled Caste students from Maharashtra",
                    "level": "State",
                    "category": ["Scholarship", "Scheduled Caste"],
                    "tags": ["SC", "Engineering", "Maharashtra"],
                },
                0.89,
            ),
            (
                {
                    "id": "br-sc",
                    "name": "Bihar SC Scholarship",
                    "details": "Scholarship for Scheduled Caste students in Bihar.",
                    "benefits": "Rs 12000",
                    "eligibility": "Scheduled Caste students from Bihar",
                    "level": "State",
                    "category": ["Scholarship", "Scheduled Caste"],
                    "tags": ["SC", "Bihar"],
                },
                0.88,
            ),
        ]

    handler = FakeHandler()
    monkeypatch.setattr(handler.rag, "search_parallel", fake_search_parallel)
    monkeypatch.setattr(server, "get_conversation_handler", lambda session_id="default": handler)

    response = await server.search_schemes(
        server.SearchRequest(
            query="scholarship for SC engineering students in Maharashtra",
            state="Maharashtra",
            category="SC",
            limit=5,
        )
    )

    names = [scheme["name"] for scheme in response["schemes"]]
    assert "Maharashtra SC Engineering Scholarship" in names
    assert "National SC Scholarship" in names
    assert "Bihar SC Scholarship" not in names
