
import sys
import numpy as np
try:
    import faiss
except ImportError as e:
    print(f"FAISS import failed: {e}")
    sys.exit(1)

print(f"Python: {sys.version}")
print(f"FAISS version: {faiss.__version__}")

# Simulate production data
n_vectors = 3400
dimension = 384
np.random.seed(1234)
embeddings = np.random.random((n_vectors, dimension)).astype('float32')

print("Normalizing...")
embeddings = np.ascontiguousarray(embeddings)
faiss.normalize_L2(embeddings)

print("Creating IVF Index...")
n_clusters = min(int(np.sqrt(n_vectors)), 100)
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)

print("Training index...")
index.train(embeddings)

print("Adding vectors...")
index.add(embeddings)

print(f"Index created with {index.ntotal} vectors")
print("FAISS production simulation passed")
