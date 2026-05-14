"""
Scholarship Voice Assistant - RAG Retrieval System
===================================================
Retrieval-Augmented Generation for scholarship search.
"""

import json
import re
import time
import pickle
import asyncio
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
from rag.query_expansion import get_query_expander  # NEW: Import query expander
from rag.semantic_cache import SemanticCache  # NEW: Import semantic cache

logger = get_logger()
config = get_config()


# Major Indian cities → state mapping for auto-location extraction
CITY_TO_STATE: dict = {
    # Uttar Pradesh
    'kanpur': 'uttar pradesh', 'lucknow': 'uttar pradesh', 'agra': 'uttar pradesh',
    'varanasi': 'uttar pradesh', 'allahabad': 'uttar pradesh', 'prayagraj': 'uttar pradesh',
    'noida': 'uttar pradesh', 'ghaziabad': 'uttar pradesh', 'meerut': 'uttar pradesh',
    'mathura': 'uttar pradesh', 'bareilly': 'uttar pradesh', 'aligarh': 'uttar pradesh',
    'moradabad': 'uttar pradesh', 'saharanpur': 'uttar pradesh', 'gorakhpur': 'uttar pradesh',
    # Maharashtra
    'mumbai': 'maharashtra', 'pune': 'maharashtra', 'nagpur': 'maharashtra',
    'nashik': 'maharashtra', 'aurangabad': 'maharashtra', 'solapur': 'maharashtra',
    'thane': 'maharashtra', 'kolhapur': 'maharashtra',
    # Karnataka
    'bangalore': 'karnataka', 'bengaluru': 'karnataka', 'mysore': 'karnataka',
    'hubli': 'karnataka', 'mangalore': 'karnataka', 'belgaum': 'karnataka',
    # Tamil Nadu
    'chennai': 'tamil nadu', 'coimbatore': 'tamil nadu', 'madurai': 'tamil nadu',
    'tiruchirappalli': 'tamil nadu', 'salem': 'tamil nadu', 'vellore': 'tamil nadu',
    # Rajasthan
    'jaipur': 'rajasthan', 'jodhpur': 'rajasthan', 'udaipur': 'rajasthan',
    'kota': 'rajasthan', 'ajmer': 'rajasthan', 'bikaner': 'rajasthan',
    # Gujarat
    'ahmedabad': 'gujarat', 'surat': 'gujarat', 'vadodara': 'gujarat',
    'rajkot': 'gujarat', 'bhavnagar': 'gujarat', 'jamnagar': 'gujarat',
    # West Bengal
    'kolkata': 'west bengal', 'calcutta': 'west bengal', 'howrah': 'west bengal',
    'durgapur': 'west bengal', 'asansol': 'west bengal', 'siliguri': 'west bengal',
    # Madhya Pradesh
    'bhopal': 'madhya pradesh', 'indore': 'madhya pradesh', 'jabalpur': 'madhya pradesh',
    'gwalior': 'madhya pradesh', 'ujjain': 'madhya pradesh', 'sagar': 'madhya pradesh',
    # Andhra Pradesh
    'visakhapatnam': 'andhra pradesh', 'vizag': 'andhra pradesh', 'vijayawada': 'andhra pradesh',
    'guntur': 'andhra pradesh', 'nellore': 'andhra pradesh', 'tirupati': 'andhra pradesh',
    # Telangana
    'hyderabad': 'telangana', 'warangal': 'telangana', 'nizamabad': 'telangana',
    # Bihar
    'patna': 'bihar', 'gaya': 'bihar', 'bhagalpur': 'bihar', 'muzaffarpur': 'bihar',
    # Punjab
    'ludhiana': 'punjab', 'amritsar': 'punjab', 'jalandhar': 'punjab', 'patiala': 'punjab',
    # Haryana
    'gurugram': 'haryana', 'gurgaon': 'haryana', 'faridabad': 'haryana',
    'panipat': 'haryana', 'ambala': 'haryana', 'rohtak': 'haryana',
    # Kerala
    'thiruvananthapuram': 'kerala', 'kochi': 'kerala', 'kozhikode': 'kerala',
    'thrissur': 'kerala', 'kollam': 'kerala',
    # Uttarakhand
    'dehradun': 'uttarakhand', 'haridwar': 'uttarakhand', 'rishikesh': 'uttarakhand',
    # Jharkhand
    'ranchi': 'jharkhand', 'jamshedpur': 'jharkhand', 'dhanbad': 'jharkhand',
    # Odisha
    'bhubaneswar': 'odisha', 'cuttack': 'odisha', 'rourkela': 'odisha',
    # Chhattisgarh
    'raipur': 'chhattisgarh', 'bhilai': 'chhattisgarh', 'bilaspur': 'chhattisgarh',
    # Assam
    'guwahati': 'assam', 'silchar': 'assam', 'dibrugarh': 'assam',
    # Delhi
    'delhi': 'delhi', 'new delhi': 'delhi',
    # Himachal Pradesh
    'shimla': 'himachal pradesh', 'manali': 'himachal pradesh',
    # Goa
    'panaji': 'goa', 'margao': 'goa', 'vasco': 'goa',
    # Jammu & Kashmir
    'srinagar': 'jammu kashmir', 'jammu': 'jammu kashmir',
}

STATE_NAMES = [
    'uttar pradesh', 'maharashtra', 'karnataka', 'tamil nadu', 'rajasthan',
    'gujarat', 'west bengal', 'madhya pradesh', 'andhra pradesh', 'bihar',
    'telangana', 'odisha', 'kerala', 'jharkhand', 'assam', 'punjab',
    'uttarakhand', 'haryana', 'himachal pradesh', 'chhattisgarh', 'manipur',
    'meghalaya', 'goa', 'arunachal pradesh', 'mizoram', 'nagaland', 'sikkim',
    'tripura', 'delhi', 'jammu kashmir', 'jammu and kashmir', 'puducherry',
    'pondicherry', 'andaman and nicobar', 'chandigarh', 'dadra and nagar haveli',
    'daman and diu', 'ladakh', 'lakshadweep',
]

STATE_ABBREVS = {
    'up': 'uttar pradesh', 'mp': 'madhya pradesh', 'tn': 'tamil nadu',
    'ap': 'andhra pradesh', 'hp': 'himachal pradesh', 'jk': 'jammu kashmir',
    'wb': 'west bengal', 'uk': 'uttarakhand',
}

# --- Fix 2: Query expansion for Indian government abbreviations ---
# Maps abbreviation (as whole word) → expanded text added to query
QUERY_EXPANSIONS = {
    'sc': 'Scheduled Caste',
    'st': 'Scheduled Tribe',
    'obc': 'Other Backward Class',
    'ews': 'Economically Weaker Section',
    'pwd': 'Person with Disability',
    'nri': 'Non Resident Indian',
    'bpl': 'Below Poverty Line',
    'apl': 'Above Poverty Line',
    'sbc': 'Special Backward Class',
    'vjnt': 'Vimukta Jati Nomadic Tribe',
    'sebc': 'Socially Educationally Backward Class',
}

# --- Fix 3: Target-group conflict detection ---
# Each group has trigger keywords (what user might type) and
# conflict keywords (what should NOT appear in results for that group)
TARGET_GROUPS = {
    'scheduled_caste': {
        'triggers': ['scheduled caste', r'\bsc\b', 'dalit'],
        'required_terms': ['scheduled caste', r'\bsc\b', 'dalit', 'scst', 'sc/st'],
        'conflict_groups': ['disability', 'women_only', 'minority', 'obc_only'],
    },
    'scheduled_tribe': {
        'triggers': ['scheduled tribe', r'\bst\b', 'tribal', 'adivasi'],
        'required_terms': ['scheduled tribe', r'\bst\b', 'tribal', 'adivasi', 'scst', 'sc/st'],
        'conflict_groups': ['disability', 'women_only', 'minority'],
    },
    'obc': {
        'triggers': [r'\bobc\b', 'other backward class', 'backward class'],
        'required_terms': [r'\bobc\b', 'other backward class', 'backward class', 'backward caste'],
        'conflict_groups': ['disability', 'women_only', 'scheduled_caste_only'],
    },
    'disability': {
        'triggers': [r'\bpwd\b', 'disabled', 'disability', 'handicap', 'divyang'],
        'required_terms': [r'\bpwd\b', 'disabled', 'disability', 'handicap', 'divyang', 'special need'],
        'conflict_groups': [],
    },
    'women': {
        'triggers': ['women', 'girl', 'female', 'widow', 'mahila'],
        'required_terms': ['women', 'girl', 'female', 'widow', 'mahila', 'lady', 'ladies'],
        'conflict_groups': ['men_only'],
    },
    'minority': {
        'triggers': ['minority', 'muslim', 'christian', 'sikh', 'buddhist', 'jain', 'parsi'],
        'required_terms': ['minority', 'muslim', 'christian', 'sikh', 'buddhist', 'jain', 'parsi'],
        'conflict_groups': ['disability'],
    },
}

# Maps conflict group name → keywords that indicate the result is for that group
CONFLICT_KEYWORDS = {
    'disability': ['persons with disability', 'person with disability', 'disabled persons',
                   'divyang', 'handicap', 'pwd', 'visually impaired', 'hearing impaired',
                   'physically challenged', 'mental retard', 'differently abled'],
    'women_only': ['only for women', 'only girls', 'female candidates only', 'exclusively for women'],
    'minority': ['minority community', 'muslim', 'christian community', 'sikh community'],
    'obc_only': ['only for obc', 'exclusively obc', 'other backward class only'],
    'scheduled_caste_only': ['only for sc', 'exclusively sc', 'scheduled caste only'],
    'men_only': ['only for men', 'only male', 'male candidates only'],
}


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
        
        # OPTIMIZATION: Semantic cache for query results
        self.semantic_cache = SemanticCache(
            similarity_threshold=0.95,  # 95% similarity required for cache hit
            max_size=500,  # Cache up to 500 queries
            ttl=3600  # 1 hour TTL
        )
        logger.info("✅ Semantic cache initialized")
    
    def load_scholarships(self, json_path: Optional[Path] = None) -> int:
        """
        Load scholarships from JSON file with deduplication.
        
        Args:
            json_path: Path to scholarships JSON file
            
        Returns:
            Number of scholarships loaded after deduplication
        """
        if json_path is None:
            json_path = config.data.scholarships_path
        
        json_path = Path(json_path)
        
        if not json_path.exists():
            logger.error(f"❌ Scholarships file not found: {json_path}")
            return 0
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_scholarships = json.load(f)
            
            logger.info(f"📚 Loaded {len(raw_scholarships)} raw scholarships from {json_path.name}")
            
            # Apply deduplication
            self.scholarships = self._deduplicate_scholarships(raw_scholarships)
            
            logger.info(f"✅ After deduplication: {len(self.scholarships)} unique scholarships")
            return len(self.scholarships)
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON: {e}")
            return 0
        except Exception as e:
            logger.error(f"❌ Failed to load scholarships: {e}")
            return 0
    
    def _deduplicate_scholarships(self, scholarships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate scholarships based on name similarity and content.
        
        Args:
            scholarships: List of scholarship dictionaries
            
        Returns:
            Deduplicated list of scholarships
        """
        if not scholarships:
            return []
        
        logger.info("🔍 Starting deduplication process...")
        
        # Step 1: Remove exact duplicates by ID
        seen_ids = set()
        unique_by_id = []
        
        for scholarship in scholarships:
            scheme_id = scholarship.get('id', '')
            if scheme_id and scheme_id in seen_ids:
                continue
            if scheme_id:
                seen_ids.add(scheme_id)
            unique_by_id.append(scholarship)
        
        logger.info(f"   After ID dedup: {len(unique_by_id)} (removed {len(scholarships) - len(unique_by_id)} exact duplicates)")
        
        # Step 2: Remove near-duplicates by name similarity
        unique_scholarships = []
        processed_names = []
        
        for scholarship in unique_by_id:
            name = scholarship.get('name', '').strip()
            if not name:
                unique_scholarships.append(scholarship)
                continue
            
            # Check if this is a near-duplicate
            is_duplicate = False
            for existing_name in processed_names:
                if self._are_names_similar(name, existing_name):
                    logger.debug(f"   Duplicate found: '{name}' ≈ '{existing_name}'")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_scholarships.append(scholarship)
                processed_names.append(name)
        
        logger.info(f"   After name dedup: {len(unique_scholarships)} (removed {len(unique_by_id) - len(unique_scholarships)} near-duplicates)")
        
        # Step 3: Remove content-based duplicates (same benefits + eligibility)
        final_scholarships = []
        content_hashes = set()
        
        for scholarship in unique_scholarships:
            content_hash = self._get_content_hash(scholarship)
            if content_hash not in content_hashes:
                final_scholarships.append(scholarship)
                content_hashes.add(content_hash)
        
        logger.info(f"   After content dedup: {len(final_scholarships)} (removed {len(unique_scholarships) - len(final_scholarships)} content duplicates)")
        
        return final_scholarships
    
    def _are_names_similar(self, name1: str, name2: str, threshold: float = 0.85) -> bool:
        """
        Check if two scholarship names are similar enough to be considered duplicates.
        
        Args:
            name1, name2: Scholarship names to compare
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if names are similar enough to be duplicates
        """
        # Normalize names for comparison
        def normalize_name(name: str) -> str:
            # Convert to lowercase
            name = name.lower()
            # Remove common prefixes/suffixes
            name = re.sub(r'^(dr\.?\s*|shri\s*|smt\.?\s*)', '', name)
            name = re.sub(r'\s*(scheme|yojana|योजना|scholarship|छात्रवृत्ति)$', '', name)
            # Remove extra whitespace and punctuation
            name = re.sub(r'[^\w\s]', ' ', name)
            name = re.sub(r'\s+', ' ', name).strip()
            return name
        
        norm1 = normalize_name(name1)
        norm2 = normalize_name(name2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return True
        
        # Jaccard similarity on words
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0
        
        return jaccard_similarity >= threshold
    
    def _get_content_hash(self, scholarship: Dict[str, Any]) -> str:
        """
        Generate a hash based on scholarship content for duplicate detection.
        
        Args:
            scholarship: Scholarship dictionary
            
        Returns:
            Content hash string
        """
        import hashlib
        
        # Extract key content fields
        benefits = str(scholarship.get('benefits', '') or scholarship.get('award_amount', '')).strip()
        eligibility = str(scholarship.get('eligibility', '')).strip()
        level = str(scholarship.get('level', '')).strip()
        state = str(scholarship.get('state', '')).strip()
        
        # Create normalized content string
        content_parts = []
        if benefits:
            # Normalize amount formats
            benefits_norm = re.sub(r'[₹,\s]', '', benefits.lower())
            content_parts.append(benefits_norm)
        
        if eligibility:
            # Normalize eligibility text
            elig_norm = re.sub(r'[^\w\s]', ' ', eligibility.lower())
            elig_norm = re.sub(r'\s+', ' ', elig_norm).strip()
            content_parts.append(elig_norm[:200])  # First 200 chars
        
        content_parts.extend([level.lower(), state.lower()])
        
        content_string = '|'.join(filter(None, content_parts))
        
        # Generate hash (re is imported at module top)
        return hashlib.md5(content_string.encode('utf-8')).hexdigest()
    
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
    
    # ------------------------------------------------------------------
    # Fix 2: Query expansion
    # ------------------------------------------------------------------
    def _expand_query(self, query: str) -> str:
        """
        ENHANCED: Expand query with Hindi synonyms and government abbreviations.
        Uses comprehensive synonym mapping for better recall.
        """
        # Step 1: Apply original government abbreviation expansion
        expanded = query
        for abbr, full in QUERY_EXPANSIONS.items():
            pattern = r'(?<![a-zA-Z])' + re.escape(abbr) + r'(?![a-zA-Z])'
            if re.search(pattern, expanded, re.IGNORECASE):
                # Append the full form if not already present
                if full.lower() not in expanded.lower():
                    expanded = re.sub(
                        pattern,
                        lambda m: m.group(0) + ' ' + full,
                        expanded,
                        flags=re.IGNORECASE,
                    )
        
        # Step 2: Apply comprehensive Hindi/synonym expansion
        query_expander = get_query_expander()
        expanded = query_expander.expand_query(expanded, max_expansions=2)
        
        if expanded != query:
            logger.info(f"🔤 Query expanded: '{query}' → '{expanded[:100]}...'")
        return expanded

    # ------------------------------------------------------------------
    # Fix 3: Target-group conflict detection
    # ------------------------------------------------------------------
    def _detect_target_group(self, query: str) -> Optional[str]:
        """Return the target group key if the query clearly targets one group."""
        q = query.lower()
        for group, cfg in TARGET_GROUPS.items():
            for trigger in cfg['triggers']:
                if re.search(trigger, q):
                    return group
        return None

    def _result_conflicts_with_group(self, doc: Dict[str, Any], target_group: str) -> bool:
        """
        Return True if this document is clearly for a DIFFERENT beneficiary group
        than what was queried.
        """
        conflict_group_keys = TARGET_GROUPS[target_group]['conflict_groups']
        if not conflict_group_keys:
            return False

        doc_text = ' '.join([
            str(doc.get('name', '')),
            str(doc.get('details', '')),
            str(doc.get('eligibility', '')),
            ' '.join(doc.get('tags', [])) if isinstance(doc.get('tags'), list) else str(doc.get('tags', '')),
            ' '.join(doc.get('category', [])) if isinstance(doc.get('category'), list) else str(doc.get('category', '')),
        ]).lower()

        for cg_key in conflict_group_keys:
            for kw in CONFLICT_KEYWORDS.get(cg_key, []):
                if kw.lower() in doc_text:
                    # Also make sure the target group itself is NOT mentioned
                    # (some schemes cover multiple groups legitimately)
                    target_terms = TARGET_GROUPS[target_group]['required_terms']
                    target_present = any(re.search(t, doc_text) for t in target_terms)
                    if not target_present:
                        return True
        return False

    # ------------------------------------------------------------------
    # Fix 1: Text-based state filter (data has no 'state' field)
    # ------------------------------------------------------------------
    def _extract_state_from_query(self, query: str) -> Optional[str]:
        """
        Auto-detect an Indian state or city in the query and return the state name.
        Checks explicit state names, abbreviations, and major city names.
        """
        q = query.lower()
        # Check full state names (longest first to avoid partial matches)
        for state in sorted(STATE_NAMES, key=len, reverse=True):
            if state in q:
                return state
        # Check state abbreviations (word-boundary match)
        words = q.split()
        for abbrev, state in STATE_ABBREVS.items():
            if abbrev in words:
                return state
        # Check city names
        for city, state in CITY_TO_STATE.items():
            if city in q:
                return state
        return None

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Hybrid search for scholarships using FAISS + BM25 + Cross-Encoder.

        Pipeline:
        1. FAISS semantic search (top 40)
        2. BM25 keyword search (top 40)
        3. Reciprocal Rank Fusion (merge results)
        4. Auto-detect location from query → apply state filter
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

        # Fix 2: Expand abbreviations in query (SC → SC Scheduled Caste, etc.)
        expanded_query = self._expand_query(query)

        # Fix 3: Detect target group for conflict filtering later
        target_group = self._detect_target_group(query)
        if target_group:
            logger.info(f"🎯 Detected target group: {target_group}")

        # Auto-detect state from query if no explicit filter provided
        if not filters:
            auto_state = self._extract_state_from_query(query)
            if auto_state:
                filters = {'state': auto_state}
                logger.info(f"📍 Auto-detected location: {auto_state}")

        # Phase 1: FAISS semantic search — use expanded query, smaller pool for speed
        t0 = time.time()
        query_embedding = self.embedding_generator.encode_query(expanded_query)
        faiss_results = self.vectorstore.search(query_embedding, top_k=8)  # Reduced from 15 to 8
        logger.latency("FAISS Search", (time.time() - t0) * 1000)

        # Phase 2: BM25 keyword search — smaller pool for speed
        t1 = time.time()
        bm25_results = self._search_bm25(expanded_query, top_k=8)  # Reduced from 15 to 8
        logger.latency("BM25 Search", (time.time() - t1) * 1000)

        # Phase 3: Reciprocal Rank Fusion
        t2 = time.time()
        fused_results = self._reciprocal_rank_fusion(faiss_results, bm25_results)
        logger.latency("RRF Fusion", (time.time() - t2) * 1000)

        fused_results = fused_results[:8]  # Reduced from 10 to 8

        # Phase 4: Text-based state filter (Fix 1)
        t3 = time.time()
        if filters:
            filtered_results = self._apply_filters(fused_results, filters)
        else:
            filtered_results = fused_results
        logger.latency("Filtering", (time.time() - t3) * 1000)

        # Ensure enough candidates; fall back to unfiltered if too few
        candidates = filtered_results if len(filtered_results) >= max(2, top_k // 2) else fused_results  # Reduced threshold
        candidates = candidates[:max(top_k, 6)]  # Smaller candidate pool

        # Phase 5: Cross-encoder re-ranking (skip for voice calls when rerank=False)
        if rerank:
            t4 = time.time()
            reranked = self._rerank_with_cross_encoder(expanded_query, candidates, top_k=top_k)
            logger.latency("Cross-Encoder Re-ranking", (time.time() - t4) * 1000)
        else:
            # Skip cross-encoder — use RRF order directly (saves 800-1500ms)
            reranked = candidates[:top_k]
            logger.info("⚡ Skipped cross-encoder reranking (voice mode)")

        # Fix 3: Remove results that conflict with the queried target group
        if target_group:
            before = len(reranked)
            reranked = [
                (doc, score) for doc, score in reranked
                if not self._result_conflicts_with_group(doc, target_group)
            ]
            removed = before - len(reranked)
            if removed:
                logger.info(f"🚫 Removed {removed} conflicting results for group '{target_group}'")

        final_results = reranked[:top_k]

        elapsed = (time.time() - overall_start) * 1000
        logger.rag_query(query, len(final_results))
        logger.latency("Total Hybrid Search", elapsed)

        return final_results

    def _build_cache_context(
        self,
        filters: Optional[Dict[str, Any]],
        rerank: bool,
        top_k: int,
    ) -> str:
        normalized_filters = json.dumps(filters or {}, sort_keys=True, ensure_ascii=False)
        return f"filters={normalized_filters}|rerank={int(rerank)}|top_k={top_k}"
    
    async def search_parallel(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Parallel hybrid search for maximum speed with semantic caching.
        Runs FAISS and BM25 searches concurrently.
        """
        if not self._is_loaded:
            logger.warning("⚠️ Index not loaded, building now...")
            if not self.build_index():
                return []

        overall_start = time.time()

        # Fix 2: Expand abbreviations in query
        expanded_query = self._expand_query(query)
        
        # OPTIMIZATION: Check semantic cache FIRST
        query_embedding = self.embedding_generator.encode_query(expanded_query)
        cache_context = self._build_cache_context(filters, rerank, top_k)
        cached_results = self.semantic_cache.get(query, query_embedding, context_key=cache_context)
        
        if cached_results is not None:
            # Cache hit! Return immediately
            elapsed = (time.time() - overall_start) * 1000
            logger.info(f"⚡ CACHE HIT - Total time: {elapsed:.0f}ms (saved ~2000ms!)")
            return cached_results[:top_k]

        # Cache miss - proceed with full search
        logger.debug(f"🔍 Cache miss - performing full search")

        # Fix 3: Detect target group for conflict filtering
        target_group = self._detect_target_group(query)
        if target_group:
            logger.info(f"🎯 Detected target group: {target_group}")

        # Auto-detect state from query if no explicit filter provided
        if not filters:
            auto_state = self._extract_state_from_query(query)
            if auto_state:
                filters = {'state': auto_state}
                logger.info(f"📍 Auto-detected location: {auto_state}")

        # Phase 1 & 2: Parallel FAISS and BM25 search
        def faiss_search():
            return self.vectorstore.search(query_embedding, top_k=8)

        def bm25_search():
            return self._search_bm25(expanded_query, top_k=8)

        t_parallel = time.time()
        
        # Run both searches concurrently
        loop = asyncio.get_event_loop()
        faiss_task = loop.run_in_executor(None, faiss_search)
        bm25_task = loop.run_in_executor(None, bm25_search)
        
        faiss_results, bm25_results = await asyncio.gather(faiss_task, bm25_task)
        
        logger.latency("Parallel Search", (time.time() - t_parallel) * 1000)

        # Phase 3: Reciprocal Rank Fusion
        t2 = time.time()
        fused_results = self._reciprocal_rank_fusion(faiss_results, bm25_results)
        logger.latency("RRF Fusion", (time.time() - t2) * 1000)

        fused_results = fused_results[:8]

        # Phase 4: Text-based state filter
        t3 = time.time()
        if filters:
            filtered_results = self._apply_filters(fused_results, filters)
        else:
            filtered_results = fused_results
        logger.latency("Filtering", (time.time() - t3) * 1000)

        # Strictly adhere to filtered results, do not fall back to fused_results
        # as that would return schemes from other states.
        candidates = filtered_results[:max(top_k, 6)]

        # Phase 5: Cross-encoder re-ranking (skip for voice calls when rerank=False)
        if rerank:
            t4 = time.time()
            reranked = self._rerank_with_cross_encoder(expanded_query, candidates, top_k=top_k)
            logger.latency("Cross-Encoder Re-ranking", (time.time() - t4) * 1000)
        else:
            reranked = candidates[:top_k]
            logger.info("⚡ Skipped cross-encoder reranking (voice mode)")

        # Fix 3: Remove results that conflict with the queried target group
        if target_group:
            before = len(reranked)
            reranked = [
                (doc, score) for doc, score in reranked
                if not self._result_conflicts_with_group(doc, target_group)
            ]
            removed = before - len(reranked)
            if removed:
                logger.info(f"🚫 Removed {removed} conflicting results for group '{target_group}'")

        final_results = reranked[:top_k]

        # OPTIMIZATION: Cache the results for future queries
        self.semantic_cache.set(query, query_embedding, final_results, context_key=cache_context)

        elapsed = (time.time() - overall_start) * 1000
        logger.rag_query(query, len(final_results))
        logger.latency("Total Parallel Search", elapsed)

        return final_results
    
    def _apply_filters(
        self,
        results: List[Tuple[Dict[str, Any], float]],
        filters: Dict[str, Any]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Fix 1: Text-based state filtering.

        The data has NO 'state' field (always None). State information is
        embedded in details/eligibility text. This method checks those text
        fields directly.

        Priority order:
        1. Central-level schemes — always relevant everywhere
        2. State schemes whose text mentions the queried state
        3. State schemes with no clear state mention — include cautiously
        4. State schemes that mention a DIFFERENT state — exclude
        """
        if not results:
            return []

        user_state = filters.get('state', '').lower().strip() if filters.get('state') else None

        central_schemes = []
        state_matched = []
        state_ambiguous = []

        for item, score in results:
            if not isinstance(item, dict):
                continue
            try:
                level = str(item.get('level', '')).lower()
                is_central = level == 'central'

                if is_central:
                    central_schemes.append((item, score))
                    continue

                if not user_state:
                    state_matched.append((item, score))
                    continue

                # Build searchable text from content fields (Fix 1 core)
                doc_text = ' '.join([
                    str(item.get('details', '')),
                    str(item.get('eligibility', '')),
                    str(item.get('name', '')),
                    str(item.get('benefits', '')),
                    ' '.join(item.get('tags', [])) if isinstance(item.get('tags'), list) else str(item.get('tags', '')),
                    ' '.join(item.get('category', [])) if isinstance(item.get('category'), list) else str(item.get('category', '')),
                ]).lower()

                if re.search(rf"\b{re.escape(user_state)}\b", doc_text):
                    state_matched.append((item, score))
                else:
                    # Check if any OTHER known state is explicitly mentioned
                    other_state_mentioned = any(
                        re.search(rf"\b{re.escape(s)}\b", doc_text)
                        for s in STATE_NAMES
                        if s != user_state
                    )
                    if other_state_mentioned:
                        pass  # Exclude — explicitly for a different state
                    else:
                        state_ambiguous.append((item, score))  # No clear state — include cautiously

            except Exception as e:
                logger.warning(f"⚠️ Error filtering item: {e}")
                continue

        # Priority: exact state match → ambiguous → central
        filtered = state_matched + state_ambiguous + central_schemes
        logger.info(
            f"📊 State filter '{user_state}': "
            f"{len(state_matched)} matched, {len(state_ambiguous)} ambiguous, "
            f"{len(central_schemes)} central → {len(filtered)} total"
        )
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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get semantic cache statistics."""
        return self.semantic_cache.get_stats()
    
    def print_cache_stats(self):
        """Print semantic cache statistics."""
        self.semantic_cache.print_stats()
    
    def clear_cache(self):
        """Clear semantic cache."""
        self.semantic_cache.clear()


# Singleton instance
_rag_instance: Optional[ScholarshipRAG] = None

def get_scholarship_rag() -> ScholarshipRAG:
    """Get the global RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = ScholarshipRAG()
    return _rag_instance
