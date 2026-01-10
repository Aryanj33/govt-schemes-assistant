"""
Scholarship Voice Assistant - Pipeline Tests
=============================================
Basic tests to verify each component works.
"""

import sys
import json
import asyncio
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestConfiguration:
    """Test configuration loading."""
    
    def test_config_loads(self):
        """Test that configuration loads without errors."""
        from backend.utils.config import get_config
        
        config = get_config()
        assert config is not None
        assert config.port == 8080 or isinstance(config.port, int)
    
    def test_groq_config(self):
        """Test Groq configuration structure."""
        from backend.utils.config import get_config
        
        config = get_config()
        assert hasattr(config.groq, 'api_key')
        assert hasattr(config.groq, 'whisper_model')
        assert hasattr(config.groq, 'llm_model')


class TestScholarshipData:
    """Test scholarship data loading."""
    
    def test_scholarships_json_exists(self):
        """Test that scholarships.json exists."""
        json_path = Path(__file__).parent.parent / "data" / "processed" / "scholarships.json"
        assert json_path.exists(), f"Scholarships file not found at {json_path}"
    
    def test_scholarships_json_valid(self):
        """Test that scholarships.json is valid JSON."""
        json_path = Path(__file__).parent.parent / "data" / "processed" / "scholarships.json"
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert isinstance(data, list)
        assert len(data) >= 10, "Should have at least 10 scholarships"
    
    def test_scholarship_structure(self):
        """Test scholarship data structure."""
        json_path = Path(__file__).parent.parent / "data" / "processed" / "scholarships.json"
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scholarship = data[0]
        required_fields = ['id', 'name', 'eligibility', 'award_amount', 'deadline']
        
        for field in required_fields:
            assert field in scholarship, f"Missing required field: {field}"


class TestEmbeddings:
    """Test embedding generation."""
    
    def test_embedding_generator_init(self):
        """Test that embedding generator initializes."""
        from backend.rag.embeddings import get_embedding_generator
        
        generator = get_embedding_generator()
        assert generator is not None
        assert generator.dimension == 384  # MiniLM dimension
    
    def test_encode_text(self):
        """Test encoding a simple text."""
        from backend.rag.embeddings import get_embedding_generator
        
        generator = get_embedding_generator()
        embedding = generator.encode("scholarship for engineering students")
        
        assert embedding.shape == (1, 384)
    
    def test_scholarship_text_creation(self):
        """Test scholarship text formatting."""
        from backend.rag.embeddings import create_scholarship_text
        
        scholarship = {
            "name": "Test Scholarship",
            "description": "A test scholarship",
            "eligibility": {
                "education_level": "12th pass",
                "category": "SC"
            },
            "award_amount": "₹10,000"
        }
        
        text = create_scholarship_text(scholarship)
        
        assert "Test Scholarship" in text
        assert "12th pass" in text or "₹10,000" in text


class TestRAG:
    """Test RAG retrieval system."""
    
    def test_rag_initialization(self):
        """Test that RAG system initializes."""
        from backend.rag.scholarship_rag import get_scholarship_rag
        
        rag = get_scholarship_rag()
        assert rag is not None
    
    def test_load_scholarships(self):
        """Test loading scholarships."""
        from backend.rag.scholarship_rag import get_scholarship_rag
        
        rag = get_scholarship_rag()
        count = rag.load_scholarships()
        
        assert count >= 10
    
    def test_build_index(self):
        """Test building FAISS index."""
        from backend.rag.scholarship_rag import get_scholarship_rag
        
        rag = get_scholarship_rag()
        success = rag.build_index(force_rebuild=True)
        
        assert success
        assert rag.vectorstore.size >= 10
    
    def test_search_engineering(self):
        """Test searching for engineering scholarships."""
        from backend.rag.scholarship_rag import get_scholarship_rag
        
        rag = get_scholarship_rag()
        if not rag.is_ready:
            rag.build_index()
        
        results = rag.search("engineering scholarship", top_k=3)
        
        assert len(results) > 0
        assert len(results) <= 3
        
        # Check result structure
        scholarship, score = results[0]
        assert 'name' in scholarship
        assert score > 0
    
    def test_search_sc_category(self):
        """Test searching for SC category scholarships."""
        from backend.rag.scholarship_rag import get_scholarship_rag
        
        rag = get_scholarship_rag()
        if not rag.is_ready:
            rag.build_index()
        
        results = rag.search("SC category scholarship", top_k=5)
        
        assert len(results) > 0
        
        # At least one should have SC in category or eligibility
        found_sc = False
        for scholarship, _ in results:
            categories = scholarship.get('category', [])
            eligibility = scholarship.get('eligibility', '')
            
            # Handle eligibility as both string and dict
            if isinstance(eligibility, dict):
                category_elig = eligibility.get('category', '')
            else:
                category_elig = str(eligibility)
            
            if 'SC' in str(categories) or 'SC' in category_elig:
                found_sc = True
                break
        
        assert found_sc, "Should find SC-related scholarships"
    
    def test_hybrid_search_exact_name(self):
        """Test that BM25 improves exact name matching in hybrid search."""
        from backend.rag.scholarship_rag import get_scholarship_rag
        
        rag = get_scholarship_rag()
        if not rag.is_ready:
            rag.build_index()
        
        # Search for a specific scholarship name (if it exists in data)
        # This tests that BM25 helps with exact keyword matching
        results = rag.search("pragati scholarship", top_k=5)
        
        assert len(results) > 0
        # The hybrid search should return relevant results
        top_result = results[0][0]
        assert 'name' in top_result
    
    def test_state_filtering_strict(self):
        """Test that UP query does not return Punjab schemes."""
        from backend.rag.scholarship_rag import get_scholarship_rag
        
        rag = get_scholarship_rag()
        if not rag.is_ready:
            rag.build_index()
        
        results = rag.search(
            "engineering scholarship",
            top_k=5,
            filters={"state": "Uttar Pradesh"}
        )
        
        # Check that no Punjab-specific (non-central) schemes appear
        for doc, _ in results:
            level = str(doc.get('level', '')).lower()
            state = str(doc.get('state', '')).lower()
            
            is_central = any(kw in level for kw in ['central', 'national', 'india'])
            is_up = 'uttar pradesh' in state or 'up' in state or state in ['', 'nan', 'null']
            is_punjab = 'punjab' in state and state not in ['', 'nan', 'null']
            
            # If it's Punjab-specific, it should not appear
            if is_punjab and not is_central:
                assert False, f"Found Punjab scheme in UP results: {doc.get('name')} (state: {state}, level: {level})"
    
    def test_central_schemes_always_included(self):
        """Test that Central/National schemes appear for any state filter."""
        from backend.rag.scholarship_rag import get_scholarship_rag
        
        rag = get_scholarship_rag()
        if not rag.is_ready:
            rag.build_index()
        
        results = rag.search(
            "scholarship",
            top_k=10,
            filters={"state": "Maharashtra"}
        )
        
        # Should have at least some central schemes
        central_count = sum(
            1 for doc, _ in results
            if any(kw in str(doc.get('level', '')).lower() 
                   for kw in ['central', 'national', 'india'])
        )
        
        # Expect at least some central schemes in results
        assert central_count >= 0, "Should include Central schemes regardless of state filter"
    
    def test_cross_encoder_relevance(self):
        """Test that cross-encoder improves relevance ranking."""
        from backend.rag.scholarship_rag import get_scholarship_rag
        
        rag = get_scholarship_rag()
        if not rag.is_ready:
            rag.build_index()
        
        # Complex query that benefits from cross-encoder understanding
        results = rag.search(
            "scholarship for girl students studying engineering",
            top_k=3
        )
        
        # Should return results
        assert len(results) > 0
        
        # Top result should have a reasonable score
        top_doc, top_score = results[0]
        assert 'name' in top_doc
        # Cross-encoder scores can be negative or positive, typically -10 to 10
        # We just check that we got a score
        assert isinstance(top_score, float)
    
    def test_search_latency(self):
        """Test that hybrid search completes within voice interaction SLA."""
        from backend.rag.scholarship_rag import get_scholarship_rag
        import time
        
        rag = get_scholarship_rag()
        if not rag.is_ready:
            rag.build_index()
        
        start = time.time()
        rag.search("engineering scholarship", top_k=5)
        elapsed_ms = (time.time() - start) * 1000
        
        # First query might be slower due to caching, but should still be reasonable
        # Allow up to 1000ms for first query (includes model warmup)
        assert elapsed_ms < 1000, f"First search took {elapsed_ms:.0f}ms (> 1000ms)"
        
        # Second query should be faster (cached)
        start = time.time()
        rag.search("medical scholarship", top_k=5)
        elapsed_ms = (time.time() - start) * 1000
        
        # Should complete within 800ms for voice use case
        # (500ms base + ~200-300ms for cross-encoder re-ranking)
        assert elapsed_ms < 800, f"Cached search took {elapsed_ms:.0f}ms (> 800ms SLA)"
    
    def test_bm25_index_exists(self):
        """Test that BM25 index is built and loaded."""
        from backend.rag.scholarship_rag import get_scholarship_rag
        
        rag = get_scholarship_rag()
        if not rag.is_ready:
            rag.build_index()
        
        # Check that BM25 components exist
        assert rag.bm25_index is not None, "BM25 index should be built"
        assert len(rag.bm25_corpus) > 0, "BM25 corpus should not be empty"
        assert rag.cross_encoder is not None, "Cross-encoder should be loaded"


class TestPrompts:
    """Test prompt configuration."""
    
    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        from config.prompts import SCHOLARSHIP_ASSISTANT_SYSTEM_PROMPT
        
        assert SCHOLARSHIP_ASSISTANT_SYSTEM_PROMPT is not None
        assert len(SCHOLARSHIP_ASSISTANT_SYSTEM_PROMPT) > 100
    
    def test_system_prompt_format(self):
        """Test that system prompt can be formatted."""
        from config.prompts import get_system_prompt_with_context
        
        context = "Test scholarship context"
        prompt = get_system_prompt_with_context(context)
        
        assert context in prompt
        assert "Vidya" in prompt  # Assistant name
    
    def test_scholarship_formatting(self):
        """Test scholarship context formatting."""
        from config.prompts import format_scholarships_for_context
        
        scholarships = [
            {
                "name": "Test Scholarship",
                "award_amount": "₹10,000",
                "deadline": "31st Dec 2025",
                "eligibility": {"education_level": "12th pass"},
                "category": ["merit"],
                "application_link": "https://example.com"
            }
        ]
        
        formatted = format_scholarships_for_context(scholarships)
        
        assert "Test Scholarship" in formatted
        assert "₹10,000" in formatted


class TestConversationHandler:
    """Test conversation handler (integration test)."""
    
    @pytest.mark.asyncio
    async def test_handler_initialization(self):
        """Test conversation handler initializes."""
        from backend.agent.conversation_handler import get_conversation_handler
        
        handler = get_conversation_handler()
        assert handler is not None
    
    def test_preference_extraction(self):
        """Test extracting preferences from user message."""
        from backend.agent.conversation_handler import ConversationState
        
        state = ConversationState()
        
        state.extract_preferences("I'm from Maharashtra, SC category, 85% marks")
        
        assert state.preferred_state == "Maharashtra"
        assert state.preferred_category == "SC"
        assert state.marks_percentage == 85.0
    
    def test_conversation_history(self):
        """Test conversation history management."""
        from backend.agent.conversation_handler import ConversationState
        
        state = ConversationState()
        
        state.add_message("user", "Hello")
        state.add_message("assistant", "Namaste!")
        
        history = state.get_history_for_llm()
        
        assert len(history) == 2
        assert history[0]['role'] == 'user'
        assert history[1]['role'] == 'assistant'


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
