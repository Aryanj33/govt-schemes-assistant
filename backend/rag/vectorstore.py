"""
Scholarship Voice Assistant - FAISS Vectorstore
================================================
Manages FAISS index for fast similarity search.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Import FAISS with fallback
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger()


class VectorStore:
    """
    FAISS-based vector store for scholarship search.
    Optimized for fast similarity search with cosine similarity.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        if not HAS_FAISS:
            raise ImportError(
                "FAISS not installed. "
                "Run: pip install faiss-cpu"
            )
        
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict[str, Any]] = []  # Stores original documents
        self._id_to_idx: Dict[str, int] = {}  # Maps document ID to index position
        
    def create_index(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """
        Create a new FAISS index from embeddings and documents.
        
        Args:
            embeddings: Numpy array of embeddings (N x dimension)
            documents: List of document dictionaries corresponding to embeddings
        """
        if len(embeddings) != len(documents):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(documents)} documents"
            )
        
        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index - using IVF for larger datasets, flat for small
        n_vectors = len(embeddings)
        
        if n_vectors < 100:
            # For small datasets, use simple flat index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product = cosine for normalized
            logger.info(f"📊 Created Flat index for {n_vectors} vectors")
        else:
            # For larger datasets, use IVF
            n_clusters = min(int(np.sqrt(n_vectors)), 100)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            logger.info(f"📊 Created IVF index with {n_clusters} clusters for {n_vectors} vectors")
        
        # Add vectors to index
        self.index.add(embeddings)
        
        # Store documents and create ID mapping
        self.documents = documents.copy()
        self._id_to_idx = {
            doc.get('id', str(i)): i 
            for i, doc in enumerate(documents)
        }
        
        logger.info(f"✅ Index created with {self.index.ntotal} vectors")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for most similar documents.
        
        Args:
            query_embedding: Query vector (1D numpy array)
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("⚠️ Index is empty, returning no results")
            return []
        
        # Prepare query
        query = np.ascontiguousarray(query_embedding.reshape(1, -1).astype(np.float32))
        faiss.normalize_L2(query)
        
        # Limit k to available vectors
        k = min(top_k, self.index.ntotal)
        
        # Set search parameters for IVF index
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(10, self.index.nlist)
        
        # Search
        scores, indices = self.index.search(query, k)
        
        # Collect results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= score_threshold:  # FAISS returns -1 for not found
                doc = self.documents[idx]
                results.append((doc, float(score)))
        
        return results
    
    def save(self, path: Path):
        """
        Save the index and documents to disk.
        
        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"💾 Saved FAISS index to {index_path}")
        
        # Save documents and metadata
        metadata = {
            "documents": self.documents,
            "dimension": self.dimension,
            "id_to_idx": self._id_to_idx
        }
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"💾 Saved metadata to {metadata_path}")
    
    def load(self, path: Path) -> bool:
        """
        Load the index and documents from disk.
        
        Args:
            path: Directory path to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        path = Path(path)
        index_path = path / "index.faiss"
        metadata_path = path / "metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            logger.warning(f"⚠️ Index files not found at {path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            logger.info(f"📥 Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.documents = metadata["documents"]
            self.dimension = metadata["dimension"]
            self._id_to_idx = metadata.get("id_to_idx", {})
            
            logger.info(f"📥 Loaded {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            return False
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """
        Add new documents to existing index.
        
        Args:
            embeddings: Numpy array of embeddings for new documents
            documents: List of new document dictionaries
        """
        if self.index is None:
            # No existing index, create new one
            self.create_index(embeddings, documents)
            return
        
        # Prepare embeddings
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        faiss.normalize_L2(embeddings)
        
        # Add to index
        start_idx = len(self.documents)
        self.index.add(embeddings)
        
        # Update documents and ID mapping
        for i, doc in enumerate(documents):
            self.documents.append(doc)
            doc_id = doc.get('id', str(start_idx + i))
            self._id_to_idx[doc_id] = start_idx + i
        
        logger.info(f"➕ Added {len(documents)} documents. Total: {self.index.ntotal}")
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID."""
        idx = self._id_to_idx.get(doc_id)
        if idx is not None and idx < len(self.documents):
            return self.documents[idx]
        return None
    
    @property
    def size(self) -> int:
        """Get the number of documents in the index."""
        return self.index.ntotal if self.index else 0
