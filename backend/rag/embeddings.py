"""
Scholarship Voice Assistant - Embedding Generator
=================================================
Uses sentence-transformers for generating vector embeddings.
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path
from functools import lru_cache

# Import sentence-transformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger()


class EmbeddingGenerator:
    """
    Generates vector embeddings using sentence-transformers.
    Uses all-MiniLM-L6-v2 model - fast, free, and good quality.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._dimension: int = 384  # Default dimension for MiniLM
        
    def _load_model(self):
        """Load the sentence-transformers model lazily."""
        if self._model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            
            logger.info(f"📥 Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"✅ Model loaded. Embedding dimension: {self._dimension}")
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            show_progress: Whether to show progress bar
            normalize: Whether to L2 normalize embeddings (recommended for cosine similarity)
            
        Returns:
            Numpy array of embeddings (N x dimension)
        """
        self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = self._model.encode(
            texts,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        Optimized for single queries.
        
        Args:
            query: Search query text
            
        Returns:
            1D numpy array of the embedding
        """
        embedding = self.encode(query, show_progress=False, normalize=True)
        return embedding[0]
    
    def encode_documents(
        self, 
        documents: List[str], 
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        Optimized for batch processing.
        
        Args:
            documents: List of document texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (N x dimension)
        """
        self._load_model()
        
        n_docs = len(documents)
        logger.info(f"📊 Generating embeddings for {n_docs} documents...")
        
        embeddings = self._model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=n_docs > 10,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings


@lru_cache(maxsize=128)
def _cached_query_hash(query: str) -> str:
    """Helper to cache query strings (hashable)."""
    return query


class CachedEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator with query caching for performance."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self._query_cache = {}
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for search query with caching.
        
        Cache hit saves ~30-50ms per query.
        """
        if query in self._query_cache:
            logger.info(f"⚡ Cache hit for query: {query[:30]}...")
            return self._query_cache[query]
        
        embedding = super().encode_query(query)
        
        # Cache the result
        if len(self._query_cache) < 100:  # Limit cache size
            self._query_cache[query] = embedding
        
        return embedding


def create_scholarship_text(item: dict) -> str:
    """
    Create a text representation of a scheme/scholarship for embedding.
    Combines all relevant fields into searchable text.
    
    Args:
        item: Dictionary containing scheme details
        
    Returns:
        Combined text string for embedding
    """
    parts = []
    
    # 1. Name & Basic Details
    name = item.get('name', '')
    parts.append(f"Scheme: {name}")
    
    # Support both old 'description' and new 'details'
    desc = item.get('details', '') or item.get('description', '')
    parts.append(desc)
    
    # 2. Benefits / Amount
    # Support new 'benefits' and old 'award_amount'
    benefits = item.get('benefits', '') or item.get('award_amount', '')
    if benefits:
        parts.append(f"Benefits: {benefits}")
        
    # 3. Eligibility
    # Support new flat 'eligibility' string and old nested 'eligibility' dict
    elig = item.get('eligibility', '')
    if isinstance(elig, dict):
        # Old nested format
        if elig.get('education_level'):
            parts.append(f"Education: {elig['education_level']}")
        if elig.get('marks_criteria'):
            parts.append(f"Marks: {elig['marks_criteria']}")
        if elig.get('category'):
            parts.append(f"Category: {elig['category']}")
        if elig.get('income_limit'):
            parts.append(f"Income: {elig['income_limit']}")
    elif elig:
        # New string format
        parts.append(f"Eligibility: {elig}")
        
    # 4. Tags / Categories / Level
    tags = item.get('tags', [])
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
        
    level = item.get('level', '')
    if level:
        parts.append(f"Level: {level}")
        
    # 5. Documents & Application (New fields)
    docs = item.get('documents', '')
    if docs:
        parts.append(f"Documents: {docs}")
        
    return " | ".join(filter(None, parts))


# Singleton instance
_embedding_generator: Optional[CachedEmbeddingGenerator] = None

def get_embedding_generator() -> CachedEmbeddingGenerator:
    """Get the global embedding generator instance with caching."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = CachedEmbeddingGenerator()
    return _embedding_generator
