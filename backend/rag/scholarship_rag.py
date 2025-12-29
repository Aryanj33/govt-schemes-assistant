"""
Scholarship Voice Assistant - RAG Retrieval System
===================================================
Retrieval-Augmented Generation for scholarship search.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        """Initialize the RAG system."""
        self.embedding_generator = get_embedding_generator()
        self.vectorstore = VectorStore(dimension=self.embedding_generator.dimension)
        self.scholarships: List[Dict[str, Any]] = []
        self._is_loaded = False
    
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
        Build or load the FAISS index.
        
        Args:
            force_rebuild: If True, rebuild index even if exists on disk
            
        Returns:
            True if index is ready, False otherwise
        """
        index_path = config.data.faiss_index_path
        
        # Try to load existing index
        if not force_rebuild and index_path.exists():
            if self.vectorstore.load(index_path):
                self._is_loaded = True
                logger.info(f"✅ Loaded existing index with {self.vectorstore.size} scholarships")
                return True
        
        # Build new index
        if not self.scholarships:
            self.load_scholarships()
        
        if not self.scholarships:
            logger.error("❌ No scholarships to index")
            return False
        
        logger.info("🔨 Building new FAISS index...")
        start_time = time.time()
        
        # Create text representations for embedding
        texts = [create_scholarship_text(s) for s in self.scholarships]
        
        # Generate embeddings
        embeddings = self.embedding_generator.encode_documents(texts)
        
        # Create index
        self.vectorstore.create_index(embeddings, self.scholarships)
        
        # Save index
        index_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save(index_path)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Index built in {elapsed:.2f}s with {self.vectorstore.size} scholarships")
        
        self._is_loaded = True
        return True
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for scholarships matching the query.
        
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
        
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_generator.encode_query(query)
        
        # Search vectorstore (get more results for filtering)
        k = top_k * 2 if filters else top_k
        results = self.vectorstore.search(query_embedding, top_k=k)
        
        # Apply filters if provided
        if filters:
            results = self._apply_filters(results, filters)
        
        # Limit to top_k
        results = results[:top_k]
        
        elapsed = (time.time() - start_time) * 1000
        logger.rag_query(query, len(results))
        logger.latency("RAG Search", elapsed)
        
        return results
    
    def _apply_filters(
        self, 
        results: List[Tuple[Dict[str, Any], float]], 
        filters: Dict[str, Any]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Apply filters to search results.
        Flexible filtering for schemes and scholarships.
        """
        filtered_results = []
        
        for item, score in results:
            if not isinstance(item, dict):
                print(f"❌ ERROR: Item is not a dict! Type: {type(item)}, Content: {item}")
                continue
                
            include = True
            
            try:
                # Get searchable text
                searchable_text = (
                    str(item.get('tags', [])) + " " + 
                    item.get('name', '') + " " + 
                    item.get('details', '') + " " +
                    str(item.get('eligibility', ''))
                ).lower()
            except Exception as e:
                print(f"❌ ERROR in _apply_filters accessing item: {e}")
                print(f"Item content: {item}")
                continue
            
            # Filter by category
            if 'category' in filters:
                required_cat = filters['category'].lower()
                # Skip if general
                if required_cat not in ['general', 'open']:
                    if required_cat not in searchable_text:
                        include = False
            
            # Filter by state
            if 'state' in filters and include:
                required_state = filters['state'].lower()
                level = str(item.get('level', '')).lower()
                
                # If central scheme, applicable to all states
                is_central = 'central' in level
                
                if not is_central:
                    if required_state not in searchable_text:
                        include = False
            
            if include:
                filtered_results.append((item, score))
        
        return filtered_results
    
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
