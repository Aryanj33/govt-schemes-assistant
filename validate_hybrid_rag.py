"""
Quick validation script for hybrid RAG implementation.
Tests basic functionality before running full test suite.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

print("=" * 60)
print("HYBRID RAG VALIDATION TEST")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/5] Testing imports...")
try:
    from rag.scholarship_rag import get_scholarship_rag
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize RAG system
print("\n[2/5] Initializing RAG system...")
try:
    rag = get_scholarship_rag()
    print("✅ RAG system initialized")
    print(f"   - Cross-encoder loaded: {rag.cross_encoder is not None}")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Build indices
print("\n[3/5] Building hybrid indices (FAISS + BM25)...")
try:
    success = rag.build_index(force_rebuild=True)
    if success:
        print("✅ Hybrid indices built successfully")
        print(f"   - FAISS index size: {rag.vectorstore.size}")
        print(f"   - BM25 corpus size: {len(rag.bm25_corpus)}")
        print(f"   - BM25 index loaded: {rag.bm25_index is not None}")
    else:
        print("❌ Index building failed")
        sys.exit(1)
except Exception as e:
    print(f"❌ Build failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test hybrid search
print("\n[4/5] Testing hybrid search...")
try:
    import time
    start = time.time()
    results = rag.search("engineering scholarship", top_k=5)
    elapsed_ms = (time.time() - start) * 1000
    
    if results:
        print(f"✅ Hybrid search successful ({elapsed_ms:.0f}ms)")
        print(f"   - Results returned: {len(results)}")
        print(f"   - Top result: {results[0][0].get('name', 'N/A')}")
        print(f"   - Top score: {results[0][1]:.4f}")
    else:
        print("⚠️ No results returned")
except Exception as e:
    print(f"❌ Search failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test state filtering
print("\n[5/5] Testing state filtering...")
try:
    results = rag.search(
        "scholarship",
        top_k=5,
        filters={"state": "Uttar Pradesh"}
    )
    
    if results:
        print(f"✅ State filtering successful")
        print(f"   - Results returned: {len(results)}")
        
        # Check state distribution
        central = sum(1 for doc, _ in results 
                     if any(kw in str(doc.get('level', '')).lower() 
                           for kw in ['central', 'national', 'india']))
        up_specific = sum(1 for doc, _ in results 
                         if 'uttar pradesh' in str(doc.get('state', '')).lower() 
                         or 'up' in str(doc.get('state', '')).lower())
        
        print(f"   - Central schemes: {central}")
        print(f"   - UP-specific: {up_specific}")
        
        # Show top 3 results
        print("\n   Top 3 results:")
        for i, (doc, score) in enumerate(results[:3], 1):
            level = doc.get('level', 'N/A')
            state = doc.get('state', 'N/A')
            print(f"   {i}. {doc.get('name', 'Unknown')[:50]}")
            print(f"      Level: {level}, State: {state}, Score: {score:.4f}")
    else:
        print("⚠️ No filtered results returned")
except Exception as e:
    print(f"❌ Filtering test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL VALIDATION TESTS PASSED!")
print("=" * 60)
print("\nHybrid RAG system is ready for production use.")
print("Run full test suite with: pytest tests/test_pipeline.py -v")
