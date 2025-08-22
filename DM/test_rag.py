#!/usr/bin/env python3
"""
Test script for RAG functionality
Tests the complete pipeline: PDF ingestion, embedding, search, and generation
"""

import os
import random
import requests
import json
from pathlib import Path
from typing import List, Dict, Any

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
DATABASE_URL = "postgresql+psycopg://yavin@localhost:5432/dm"
OLLAMA_MODEL = "phi3:latest"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        # Test generate endpoint
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama running with {len(models)} models:")
            for model in models:
                print(f"   - {model['name']}")
            return True
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

def test_ollama_generate():
    """Test basic text generation"""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": "Say hello in one sentence.",
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generation test successful: {result.get('response', '')[:100]}...")
            return True
        else:
            print(f"❌ Generation failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Generation test error: {e}")
        return False

def test_ollama_embeddings():
    """Test embedding generation"""
    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": OLLAMA_EMBED_MODEL,
                "prompt": "test text"
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            embedding = result.get("embedding") or result.get("embeddings") or result.get("data", [{}])[0].get("embedding")
            if embedding:
                print(f"✅ Embedding test successful: {len(embedding)} dimensions")
                return True
            else:
                print("❌ No embedding returned")
                return False
        else:
            print(f"❌ Embedding failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Embedding test error: {e}")
        return False

def test_database_connection():
    """Test database connectivity"""
    try:
        import psycopg
        conn = psycopg.connect("postgresql://yavin@localhost:5432/dm")
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Database connected: {version[0][:50]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_pgvector_extension():
    """Test if pgvector extension is available"""
    try:
        import psycopg
        conn = psycopg.connect("postgresql://yavin@localhost:5432/dm")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        result = cursor.fetchone()
        if result:
            print("✅ pgvector extension is available")
            cursor.close()
            conn.close()
            return True
        else:
            print("❌ pgvector extension not found")
            cursor.close()
            conn.close()
            return False
    except Exception as e:
        print(f"❌ pgvector test failed: {e}")
        return False

def test_rag_store():
    """Test RAG store functionality"""
    try:
        from rag_store import RagStore
        import os
        # Set the correct database URL for testing
        os.environ["DATABASE_URL"] = "postgresql+psycopg://yavin@localhost:5432/dm"
        store = RagStore()
        store.ensure_schema()
        print("✅ RAG store initialized successfully")
        return True
    except Exception as e:
        print(f"❌ RAG store test failed: {e}")
        return False

def find_random_pdf():
    """Find a random PDF file for testing"""
    docs_dir = Path("/Users/yavin/python_projects/DataManagement_Assistant/DM/it-management-and-audit-source-main")
    if not docs_dir.exists():
        print(f"❌ Documents directory not found: {docs_dir}")
        return None
    
    pdf_files = list(docs_dir.rglob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found")
        return None
    
    selected_pdf = random.choice(pdf_files)
    print(f"📄 Selected test PDF: {selected_pdf.name}")
    return str(selected_pdf)

def test_complete_rag_pipeline(pdf_path: str):
    """Test the complete RAG pipeline"""
    try:
        from rag_store import RagStore
        from ollama import embed_texts
        import os
        
        print(f"\n🔄 Testing complete RAG pipeline with {os.path.basename(pdf_path)}")
        
        # Set the correct database URL for testing
        os.environ["DATABASE_URL"] = "postgresql+psycopg://yavin@localhost:5432/dm"
        
        # Initialize store
        store = RagStore()
        store.ensure_schema()
        
        # Test ingestion
        print("   📥 Ingesting PDF...")
        added, changed = store.ingest_pdf(pdf_path, embedder=embed_texts)
        print(f"   ✅ Ingested {added} chunks ({'updated' if changed else 'cached'})")
        
        # Test search
        print("   🔍 Testing search...")
        test_question = "What is the main topic of this document?"
        q_emb = embed_texts([test_question])[0]
        results = store.search(q_emb, document_paths=[pdf_path], k=3)
        print(f"   ✅ Search returned {len(results)} results")
        
        # Test generation
        print("   🤖 Testing generation...")
        context = "\n\n".join(f"[From {r.document_title}]\n{r.text}" for r in results[:2])
        prompt = f"Context:\n{context}\n\nQuestion: {test_question}\nAnswer based on this context:"
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "")[:200]
            print(f"   ✅ Generation successful: {answer}...")
            return True
        else:
            print(f"   ❌ Generation failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ RAG pipeline test failed: {e}")
        return False

def test_streamlit_endpoint():
    """Test if Streamlit app is responding"""
    try:
        response = requests.get("http://127.0.0.1:8501/", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit app is responding")
            return True
        else:
            print(f"❌ Streamlit responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Streamlit: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RAG System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Ollama Generation", test_ollama_generate),
        ("Ollama Embeddings", test_ollama_embeddings),
        ("Database Connection", test_database_connection),
        ("pgvector Extension", test_pgvector_extension),
        ("RAG Store", test_rag_store),
        ("Streamlit Endpoint", test_streamlit_endpoint),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ Test crashed: {e}")
            results[test_name] = False
    
    # Test complete pipeline if basic tests pass
    if all(results.values()):
        print(f"\n🎯 All basic tests passed! Testing complete RAG pipeline...")
        pdf_path = find_random_pdf()
        if pdf_path:
            results["Complete RAG Pipeline"] = test_complete_rag_pipeline(pdf_path)
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 50)
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RAG system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
