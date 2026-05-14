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
    Create a high-quality text representation of a scheme/scholarship for embedding.
    IMPROVED: Better text structure, Hindi support, and keyword optimization.
    
    Args:
        item: Dictionary containing scheme details
        
    Returns:
        Combined text string optimized for semantic search
    """
    parts = []
    
    # 1. Name & Basic Details - IMPROVED with keyword variations
    name = item.get('name', '')
    if name:
        parts.append(f"Scheme Name: {name}")
        
        # Add common keyword variations for better matching
        name_lower = name.lower()
        if 'scholarship' in name_lower or 'छात्रवृत्ति' in name_lower:
            parts.append("Type: Scholarship छात्रवृत्ति Student Financial Aid")
        elif 'kisan' in name_lower or 'farmer' in name_lower or 'किसान' in name_lower:
            parts.append("Type: Farmer Scheme किसान योजना Agriculture")
        elif 'business' in name_lower or 'loan' in name_lower or 'व्यापार' in name_lower:
            parts.append("Type: Business Loan व्यापार ऋण Entrepreneurship")
    
    # 2. Description/Details - IMPROVED with better formatting
    desc = item.get('details', '') or item.get('description', '')
    if desc:
        # Clean and format description
        desc_clean = desc.strip()
        if len(desc_clean) > 500:
            # Truncate very long descriptions but keep key info
            desc_clean = desc_clean[:500] + "..."
        parts.append(f"Description: {desc_clean}")
    
    # 3. Benefits/Amount - IMPROVED with standardized format
    benefits = item.get('benefits', '') or item.get('award_amount', '')
    if benefits:
        benefits_clean = str(benefits).strip()
        # Standardize amount format for better matching
        if '₹' in benefits_clean or 'rupees' in benefits_clean.lower():
            parts.append(f"Financial Benefits: {benefits_clean}")
        else:
            parts.append(f"Benefits: {benefits_clean}")
    
    # 4. Eligibility - IMPROVED with structured extraction
    elig = item.get('eligibility', '')
    if isinstance(elig, dict):
        # Old nested format - extract key criteria
        elig_parts = []
        if elig.get('education_level'):
            elig_parts.append(f"Education: {elig['education_level']}")
        if elig.get('marks_criteria'):
            elig_parts.append(f"Minimum Marks: {elig['marks_criteria']}")
        if elig.get('category'):
            cat = elig['category']
            # Add Hindi equivalents for categories
            if cat == 'SC':
                elig_parts.append("Category: SC Scheduled Caste अनुसूचित जाति")
            elif cat == 'ST':
                elig_parts.append("Category: ST Scheduled Tribe अनुसूचित जनजाति")
            elif cat == 'OBC':
                elig_parts.append("Category: OBC Other Backward Class अन्य पिछड़ा वर्ग")
            elif cat == 'General':
                elig_parts.append("Category: General सामान्य")
            else:
                elig_parts.append(f"Category: {cat}")
        if elig.get('income_limit'):
            elig_parts.append(f"Income Limit: {elig['income_limit']}")
        
        if elig_parts:
            parts.append("Eligibility: " + " | ".join(elig_parts))
    elif elig:
        # New string format - clean and structure
        elig_clean = str(elig).strip()
        if len(elig_clean) > 300:
            # Extract key eligibility points
            elig_clean = elig_clean[:300] + "..."
        parts.append(f"Eligibility Criteria: {elig_clean}")
    
    # 5. State/Level - IMPROVED with geographic keywords
    state = item.get('state', '')
    level = item.get('level', '')
    
    if level:
        level_clean = str(level).strip()
        if 'central' in level_clean.lower() or 'national' in level_clean.lower():
            parts.append("Scope: Central National All India सभी राज्य")
        elif 'state' in level_clean.lower():
            parts.append(f"Scope: State Level राज्य स्तर {state}")
        else:
            parts.append(f"Level: {level_clean}")
    
    if state and state.lower() not in ['nan', 'null', '']:
        # Add Hindi state names for better matching
        state_hindi_map = {
            'Uttar Pradesh': 'उत्तर प्रदेश यूपी UP',
            'Maharashtra': 'महाराष्ट्र',
            'Bihar': 'बिहार',
            'West Bengal': 'पश्चिम बंगाल',
            'Rajasthan': 'राजस्थान',
            'Karnataka': 'कर्नाटक',
            'Gujarat': 'गुजरात',
            'Madhya Pradesh': 'मध्य प्रदेश MP',
            'Tamil Nadu': 'तमिल नाडु',
            'Kerala': 'केरल',
            'Punjab': 'पंजाब',
            'Haryana': 'हरियाणा',
            'Delhi': 'दिल्ली',
        }
        
        hindi_name = state_hindi_map.get(state, '')
        if hindi_name:
            parts.append(f"State: {state} {hindi_name}")
        else:
            parts.append(f"State: {state}")
    
    # 6. Target Groups - IMPROVED with demographic keywords
    categories = item.get('categories', []) or item.get('category', [])
    if categories:
        if isinstance(categories, str):
            categories = [categories]
        
        # Add demographic keywords for better matching
        demo_keywords = []
        for cat in categories:
            cat_lower = str(cat).lower()
            if 'women' in cat_lower or 'girl' in cat_lower:
                demo_keywords.append("Women Girls महिला लड़की Female")
            elif 'minority' in cat_lower:
                demo_keywords.append("Minority अल्पसंख्यक Muslim Christian Sikh")
            elif 'disabled' in cat_lower or 'divyang' in cat_lower:
                demo_keywords.append("Disabled Divyang दिव्यांग Handicapped")
            elif 'merit' in cat_lower:
                demo_keywords.append("Merit Based योग्यता आधारित")
        
        if demo_keywords:
            parts.append(f"Target Groups: {' '.join(demo_keywords)}")
        
        parts.append(f"Categories: {', '.join(str(c) for c in categories)}")
    
    # 7. Tags - IMPROVED with keyword expansion
    tags = item.get('tags', [])
    if tags:
        # Expand tags with related keywords
        expanded_tags = []
        for tag in tags:
            tag_str = str(tag).lower()
            expanded_tags.append(str(tag))
            
            # Add related keywords
            if 'engineering' in tag_str:
                expanded_tags.append("BTech B.Tech Technical बीटेक इंजीनियरिंग")
            elif 'medical' in tag_str:
                expanded_tags.append("MBBS Doctor Medicine चिकित्सा डॉक्टर")
            elif 'agriculture' in tag_str:
                expanded_tags.append("Farming खेती कृषि Krishi")
            elif 'education' in tag_str:
                expanded_tags.append("Study पढ़ाई शिक्षा Learning")
        
        parts.append(f"Keywords: {' '.join(expanded_tags)}")
    
    # 8. Documents - IMPROVED with application keywords
    docs = item.get('documents', '')
    if docs:
        parts.append(f"Required Documents: {docs}")
    
    # 9. Application Process - NEW
    app_link = item.get('applicationLink', '') or item.get('application_link', '')
    if app_link and 'http' in app_link:
        parts.append("Application: Online Available ऑनलाइन आवेदन")
    
    # Join with improved separators for better semantic understanding
    return " || ".join(filter(None, parts))


# Singleton instance
_embedding_generator: Optional[CachedEmbeddingGenerator] = None

def get_embedding_generator() -> CachedEmbeddingGenerator:
    """Get the global embedding generator instance with caching."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = CachedEmbeddingGenerator()
    return _embedding_generator
