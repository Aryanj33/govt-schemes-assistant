"""
Scholarship Voice Assistant - RAG Retrieval System
===================================================
Retrieval-Augmented Generation for scholarship search.
"""

import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from utils.logger import get_logger
from utils.config import get_config
from rag.embeddings import get_embedding_generator, create_scholarship_text
from rag.vectorstore import VectorStore

logger = get_logger()
config = get_config()


class ScholarshipRAG:
    """
    RAG system for scholarship search and retrieval.
    Handles loading data, indexing, and semantic search.
    """
    
    def __init__(self):
        """Initialize the RAG system with hybrid search capabilities."""
        self.embedding_generator = get_embedding_generator()
        self.vectorstore = VectorStore(dimension=self.embedding_generator.dimension)
        self.scholarships: List[Dict[str, Any]] = []
        self._is_loaded = False
        
        # BM25 index for keyword-based search
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []  # Tokenized documents
        
        # Cross-encoder for re-ranking (load once at initialization)
        logger.info("📥 Loading cross-encoder model for re-ranking...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("✅ Cross-encoder loaded")
    
    def load_scholarships(self, json_path: Optional[Path] = None) -> int:
        """
        Load scholarships from JSON file.
        
        Args:
            json_path: Path to scholarships JSON file
            
        Returns:
            Number of scholarships loaded
        """
        if json_path is None:
            json_path = config.data.scholarships_path
        
        json_path = Path(json_path)
        
        if not json_path.exists():
            logger.error(f"❌ Scholarships file not found: {json_path}")
            return 0
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.scholarships = json.load(f)
            
            logger.info(f"📚 Loaded {len(self.scholarships)} scholarships from {json_path.name}")
            return len(self.scholarships)
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON: {e}")
            return 0
        except Exception as e:
            logger.error(f"❌ Failed to load scholarships: {e}")
            return 0
    
    def build_index(self, force_rebuild: bool = False) -> bool:
        """
        Build or load the FAISS and BM25 indices.
        
        Args:
            force_rebuild: If True, rebuild index even if exists on disk
            
        Returns:
            True if index is ready, False otherwise
        """
        index_path = config.data.faiss_index_path
        bm25_path = index_path / "bm25_index.pkl"
        
        # Try to load existing indices
        if not force_rebuild and index_path.exists():
            if self.vectorstore.load(index_path):
                # Try to load BM25 index
                if bm25_path.exists():
                    try:
                        with open(bm25_path, 'rb') as f:
                            bm25_data = pickle.load(f)
                        self.bm25_index = bm25_data['index']
                        self.bm25_corpus = bm25_data['corpus']
                        logger.info(f"✅ Loaded BM25 index with {len(self.bm25_corpus)} documents")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load BM25 index: {e}. Will rebuild.")
                        force_rebuild = True
                
                if not force_rebuild:
                    self._is_loaded = True
                    logger.info(f"✅ Loaded existing indices with {self.vectorstore.size} scholarships")
                    return True
        
        # Build new indices
        if not self.scholarships:
            self.load_scholarships()
        
        if not self.scholarships:
            logger.error("❌ No scholarships to index")
            return False
        
        logger.info("🔨 Building new FAISS and BM25 indices...")
        start_time = time.time()
        
        # Create text representations for embedding
        texts = [create_scholarship_text(s) for s in self.scholarships]
        
        # Generate embeddings for FAISS
        embeddings = self.embedding_generator.encode_documents(texts)
        
        # Create FAISS index
        self.vectorstore.create_index(embeddings, self.scholarships)
        
        # Build BM25 index
        logger.info("🔨 Building BM25 keyword index...")
        # Tokenize documents (simple whitespace + lowercase)
        self.bm25_corpus = [text.lower().split() for text in texts]
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        logger.info(f"✅ BM25 index built with {len(self.bm25_corpus)} documents")
        
        # Save indices
        index_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save(index_path)
        
        # Save BM25 index
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'index': self.bm25_index,
                'corpus': self.bm25_corpus
            }, f)
        logger.info(f"💾 Saved BM25 index to {bm25_path}")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Hybrid indices built in {elapsed:.2f}s with {self.vectorstore.size} scholarships")
        
        self._is_loaded = True
        return True
    
    def _search_bm25(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Perform BM25 keyword-based search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of (doc_index, bm25_score) tuples
        """
        if self.bm25_index is None:
            logger.warning("⚠️ BM25 index not available")
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Return as (index, score) tuples
        return [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    def _reciprocal_rank_fusion(
        self,
        faiss_results: List[Tuple[Dict[str, Any], float]],
        bm25_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Combine FAISS and BM25 results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score(d) = Σ 1 / (k + rank(d))
        
        Args:
            faiss_results: List of (document, faiss_score) tuples
            bm25_results: List of (doc_index, bm25_score) tuples
            k: Constant for RRF (default 60, standard value)
            
        Returns:
            Fused list of (document, rrf_score) tuples, sorted by RRF score
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        
        # Process FAISS results
        for rank, (doc, _) in enumerate(faiss_results):
            doc_id = doc.get('id', str(id(doc)))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
            doc_map[doc_id] = doc
        
        # Process BM25 results
        for rank, (idx, _) in enumerate(bm25_results):
            if idx < len(self.scholarships):
                doc = self.scholarships[idx]
                doc_id = doc.get('id', str(id(doc)))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
                doc_map[doc_id] = doc
        
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return as (document, score) tuples
        return [(doc_map[doc_id], score) for doc_id, score in sorted_docs]
    
    def _rerank_with_cross_encoder(
        self,
        query: str,
        candidates: List[Tuple[Dict[str, Any], float]],
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Re-rank top candidates using Cross-Encoder for precise relevance.
        
        Args:
            query: User's natural language query
            candidates: List of (scholarship, hybrid_score) tuples
            top_k: Number of final results to return
            
        Returns:
            Re-ranked list of top_k results with cross-encoder scores
        """
        if not candidates:
            return []
        
        # Prepare query-document pairs
        pairs = [
            [query, create_scholarship_text(doc)]
            for doc, _ in candidates
        ]
        
        # Get cross-encoder scores
        ce_scores = self.cross_encoder.predict(pairs)
        
        # Combine with original documents
        reranked = [
            (candidates[i][0], float(ce_scores[i]))
            for i in range(len(candidates))
        ]
        
        # Sort by cross-encoder score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Hybrid search for scholarships using FAISS + BM25 + Cross-Encoder.
        
        Pipeline:
        1. FAISS semantic search (top 20)
        2. BM25 keyword search (top 20)
        3. Reciprocal Rank Fusion (merge results)
        4. Metadata filtering (state, category)
        5. Cross-encoder re-ranking (final top_k)
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            filters: Optional filters (category, state, etc.)
            
        Returns:
            List of (scholarship, score) tuples
        """
        if not self._is_loaded:
            logger.warning("⚠️ Index not loaded, building now...")
            if not self.build_index():
                return []
        
        overall_start = time.time()
        
        # Phase 1: FAISS semantic search
        t0 = time.time()
        query_embedding = self.embedding_generator.encode_query(query)
        faiss_results = self.vectorstore.search(query_embedding, top_k=20)
        logger.latency("FAISS Search", (time.time() - t0) * 1000)
        
        # Phase 2: BM25 keyword search
        t1 = time.time()
        bm25_results = self._search_bm25(query, top_k=20)
        logger.latency("BM25 Search", (time.time() - t1) * 1000)
        
        # Phase 3: Reciprocal Rank Fusion
        t2 = time.time()
        fused_results = self._reciprocal_rank_fusion(faiss_results, bm25_results)
        logger.latency("RRF Fusion", (time.time() - t2) * 1000)
        
        # Take top 15 from fusion for filtering
        fused_results = fused_results[:15]
        
        # Phase 4: Apply metadata filters
        t3 = time.time()
        if filters:
            filtered_results = self._apply_filters(fused_results, filters)
        else:
            filtered_results = fused_results
        logger.latency("Filtering", (time.time() - t3) * 1000)
        
        # --- START OF HIGH-CONFIDENCE BYPASS ---
        BYPASS_THRESHOLD = 0.03  # RRF score threshold (top 5 in both FAISS & BM25)
        if filtered_results and filtered_results[0][1] >= BYPASS_THRESHOLD:
            logger.info(f"⚡ High-confidence bypass triggered (Score: {filtered_results[0][1]:.4f})")
            final_results = filtered_results[:top_k]
            
            # Log metrics and exit early
            elapsed = (time.time() - overall_start) * 1000
            logger.rag_query(query, len(final_results))
            logger.latency("Total Hybrid Search (BYPASSED)", elapsed)
            return final_results
        # --- END OF BYPASS ---
        
        # Take top 10 for re-ranking
        candidates = filtered_results[:10]
        
        # Phase 5: Cross-encoder re-ranking
        t4 = time.time()
        final_results = self._rerank_with_cross_encoder(query, candidates, top_k=top_k)
        logger.latency("Cross-Encoder Re-ranking", (time.time() - t4) * 1000)
        
        # Log overall metrics
        elapsed = (time.time() - overall_start) * 1000
        logger.rag_query(query, len(final_results))
        logger.latency("Total Hybrid Search", elapsed)
        
        # Log retrieval precision
        if filters and 'state' in filters:
            state_matched = sum(
                1 for doc, _ in final_results
                if filters['state'].lower() in str(doc.get('state', '')).lower()
                or 'central' in str(doc.get('level', '')).lower()
                or 'national' in str(doc.get('level', '')).lower()
            )
            precision = state_matched / len(final_results) if final_results else 0
            logger.info(f"🎯 State Filter Precision: {precision:.1%} ({state_matched}/{len(final_results)} matched)")
        
        return final_results
    
    def _apply_filters(
        self, 
        results: List[Tuple[Dict[str, Any], float]], 
        filters: Dict[str, Any]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Apply robust metadata filters with Central scheme prioritization.
        
        Priority order:
        1. Central/National schemes (always included - applicable everywhere)
        2. State-specific schemes matching user's state
        3. Strictly exclude schemes from other states
        """
        if not results:
            return []
        
        user_state = filters.get('state', '').lower() if filters.get('state') else None
        user_category = filters.get('category', '').lower() if filters.get('category') else None
        
        central_schemes = []
        state_matched = []
        
        for item, score in results:
            if not isinstance(item, dict):
                logger.warning(f"⚠️ Skipping non-dict item in filter: {type(item)}")
                continue
            
            try:
                level = str(item.get('level', '')).lower()
                scheme_state = str(item.get('state', '')).lower()
                
                # Check if Central/National scheme (applicable to all states)
                is_central = any(
                    keyword in level 
                    for keyword in ['central', 'national', 'india', 'all india', 'pan india']
                )
                
                # Central schemes always included
                if is_central:
                    central_schemes.append((item, score))
                    continue
                
                # State filtering (strict for non-central schemes)
                if user_state:
                    # Scheme has explicit state field
                    if scheme_state and scheme_state not in ['nan', 'null', 'none', '']:
                        # Check for state match (handle variations like "uttar pradesh" vs "up")
                        state_match = (
                            user_state in scheme_state or
                            scheme_state in user_state or
                            self._state_variants_match(user_state, scheme_state)
                        )
                        
                        if state_match:
                            state_matched.append((item, score))
                        # else: Exclude (different state)
                    else:
                        # No explicit state - check if it's actually state-level
                        if 'state' in level:
                            # It claims to be state-level but no state specified - risky, skip
                            continue
                        else:
                            # Might be general/ambiguous - include cautiously
                            state_matched.append((item, score))
                else:
                    # No state filter - include all non-central
                    state_matched.append((item, score))
                
                # Category filtering (relaxed due to data quality issues)
                # Note: Keeping this light as the cross-encoder will handle relevance
                if user_category and user_category not in ['general', 'open']:
                    # For now, defer to cross-encoder rather than strict category filtering
                    # Future: Implement strict category logic when data is cleaned
                    pass
                    
            except Exception as e:
                logger.warning(f"⚠️ Error filtering item: {e}")
                continue
        
        # Merge: State-matched first (higher priority), then Central schemes
        filtered = state_matched + central_schemes
        
        logger.info(f"📊 Filter results: {len(state_matched)} state, {len(central_schemes)} central, {len(filtered)} total")
        
        return filtered
    
    def _state_variants_match(self, state1: str, state2: str) -> bool:
        """
        Check if two state strings match considering common abbreviations.
        
        Examples:
        - "uttar pradesh" <-> "up"
        - "tamil nadu" <-> "tn"
        """
        # Common state abbreviations
        variants = {
            'up': 'uttar pradesh',
            'mp': 'madhya pradesh',
            'tn': 'tamil nadu',
            'ap': 'andhra pradesh',
            'hp': 'himachal pradesh',
            'jk': 'jammu kashmir',
            'wb': 'west bengal',
        }
        
        s1 = state1.strip()
        s2 = state2.strip()
        
        # Check direct variants
        if s1 in variants and variants[s1] == s2:
            return True
        if s2 in variants and variants[s2] == s1:
            return True
        
        return False
    
    def format_for_llm(self, results: List[Tuple[Dict[str, Any], float]]) -> str:
        """Format search results for LLM context."""
        if not results:
            return "No relevant government schemes found."
            
        context_parts = ["Found the following relevant government schemes:"]
        
        for i, (item, score) in enumerate(results, 1):
            if not isinstance(item, dict):
                logger.error(f"❌ format_for_llm encountered non-dict item: {type(item)}")
                continue
                
            try:
                name = item.get('name', 'Unknown Scheme')
                details = item.get('details', 'No details available')
                benefits = item.get('benefits', 'No specific benefits listed')
                eligibility = item.get('eligibility', 'No eligibility criteria listed')
                app_process = item.get('application_process', 'No application process listed')
                docs = item.get('documents', 'No documents listed')
                source = item.get('source', 'Government of India')
                
                context_parts.append(f"""
{i}. {name} (Relevance: {score:.2f})
   Details: {details}
   Benefits: {benefits}
   Eligibility: {eligibility}
   Application: {app_process}
   Documents: {docs}
   Source: {source}
""")
            except Exception as e:
                logger.error(f"❌ Error formatting item {i}: {e}")
                continue
        
        return "\n".join(context_parts)
    
    def get_scholarship_by_id(self, scholarship_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific scholarship by its ID."""
        return self.vectorstore.get_document_by_id(scholarship_id)
    
    @property
    def is_ready(self) -> bool:
        """Check if the RAG system is ready for queries."""
        return self._is_loaded and self.vectorstore.size > 0


# Singleton instance
_rag_instance: Optional[ScholarshipRAG] = None

def get_scholarship_rag() -> ScholarshipRAG:
    """Get the global RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = ScholarshipRAG()
    return _rag_instance
