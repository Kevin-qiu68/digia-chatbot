"""
Test script for RAG functionality
Run this to test the RAG retrieval and generation pipeline
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import (
    COHERE_API_KEY,
    VECTORDB_PATH,
    COLLECTION_NAME,
    COHERE_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_RERANK
)
from src.vectorstore import VectorStoreManager
from src.rag_chain import RAGChain


def test_basic_retrieval(rag_chain: RAGChain):
    """Test basic semantic retrieval"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Semantic Retrieval")
    print("=" * 60)
    
    query = "What does Digia do?"
    print(f"\nQuery: {query}")
    
    docs = rag_chain.retrieve_documents(query)
    
    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n{i}. Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"   Content: {doc.page_content[:150]}...")


def test_reranking(rag_chain: RAGChain):
    """Test reranking functionality"""
    print("\n" + "=" * 60)
    print("TEST 2: Reranking")
    print("=" * 60)
    
    query = "Tell me about Digia's services"
    print(f"\nQuery: {query}")
    
    # Retrieve documents
    docs = rag_chain.retrieve_documents(query)
    
    # Rerank
    reranked_docs = rag_chain.rerank_documents(query, docs)
    
    print(f"\nTop {len(reranked_docs)} reranked documents:")
    for i, doc in enumerate(reranked_docs, 1):
        print(f"\n{i}. Relevance Score: {doc['relevance_score']:.3f}")
        print(f"   Source: {doc['metadata'].get('source', 'Unknown')}")
        print(f"   Content: {doc['content'][:150]}...")


def test_full_rag_pipeline(rag_chain: RAGChain):
    """Test complete RAG pipeline"""
    print("\n" + "=" * 60)
    print("TEST 3: Full RAG Pipeline (with Rerank)")
    print("=" * 60)
    
    queries = [
        "What is Digia?",
        "What services does Digia provide?",
        "How can I contact Digia?",
        "Tell me about Digia's products"
    ]
    
    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print('=' * 60)
        
        result = rag_chain.query(query, use_rerank=True)
        
        if result['error']:
            print(f"\n❌ Error: {result['error']}")
            continue
        
        print(f"\n📝 Answer:")
        print(result['answer'])
        
        print(f"\n📚 Sources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['source']} (Relevance: {source['relevance_score']:.3f})")


def test_without_rerank(rag_chain: RAGChain):
    """Test RAG pipeline without reranking"""
    print("\n" + "=" * 60)
    print("TEST 4: RAG Pipeline (without Rerank)")
    print("=" * 60)
    
    query = "What is Digia?"
    print(f"\nQuery: {query}")
    
    result = rag_chain.query(query, use_rerank=False)
    
    print(f"\n📝 Answer:")
    print(result['answer'])
    
    print(f"\n📚 Sources ({len(result['sources'])}):")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['source']}")


def interactive_mode(rag_chain: RAGChain):
    """Interactive testing mode"""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Type your questions (or 'quit' to exit)")
    
    while True:
        query = input("\n\nYour question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Exiting interactive mode...")
            break
        
        if not query:
            continue
        
        result = rag_chain.query(query, use_rerank=True)
        
        print(f"\n{'=' * 60}")
        print("Answer:")
        print('=' * 60)
        print(result['answer'])
        
        print(f"\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['source']} (Score: {source['relevance_score']:.3f})")


def main():
    """Main test function"""
    print("\n" + "=" * 70)
    print("RAG SYSTEM TEST SUITE")
    print("=" * 70)
    
    # Check API key
    if not COHERE_API_KEY:
        print("❌ Error: COHERE_API_KEY not found in .env file")
        return
    
    # Check if vector database exists
    if not os.path.exists(VECTORDB_PATH):
        print(f"❌ Error: Vector database not found at {VECTORDB_PATH}")
        print("Please run build_vectordb.py first")
        return
    
    try:
        # Initialize RAG chain
        print("\nInitializing RAG Chain...")
        
        vectorstore_manager = VectorStoreManager(
            api_key=COHERE_API_KEY,
            persist_directory=VECTORDB_PATH,
            collection_name=COLLECTION_NAME
        )
        
        rag_chain = RAGChain(
            vectorstore_manager=vectorstore_manager,
            cohere_api_key=COHERE_API_KEY,
            model=COHERE_MODEL,
            top_k_retrieval=TOP_K_RETRIEVAL,
            top_k_rerank=TOP_K_RERANK
        )
        
        print("✓ RAG Chain initialized successfully")
        
        # Run tests
        print("\n" + "=" * 70)
        print("Select test mode:")
        print("  1. Run all automated tests")
        print("  2. Test basic retrieval only")
        print("  3. Test reranking only")
        print("  4. Test full RAG pipeline")
        print("  5. Interactive mode")
        print("=" * 70)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            test_basic_retrieval(rag_chain)
            test_reranking(rag_chain)
            test_full_rag_pipeline(rag_chain)
            test_without_rerank(rag_chain)
        elif choice == '2':
            test_basic_retrieval(rag_chain)
        elif choice == '3':
            test_reranking(rag_chain)
        elif choice == '4':
            test_full_rag_pipeline(rag_chain)
        elif choice == '5':
            interactive_mode(rag_chain)
        else:
            print("Invalid choice. Running all tests...")
            test_basic_retrieval(rag_chain)
            test_reranking(rag_chain)
            test_full_rag_pipeline(rag_chain)
        
        print("\n" + "=" * 70)
        print("✅ Testing Complete!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()