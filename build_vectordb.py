"""
Main script for building vector database
Run this script to create or update the vector database
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import (
    COHERE_API_KEY, 
    VECTORDB_PATH, 
    COLLECTION_NAME,
    DATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from src.data_loader import DocumentLoader
from src.vectorstore import VectorStoreManager


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("Digia AI Assistant - Vector Database Builder")
    print("=" * 60)
    
    # Check API key
    if not COHERE_API_KEY:
        print("❌ Error: COHERE_API_KEY not found")
        print("Please set your Cohere API key in the .env file")
        return
    
    # Check data directory
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data directory does not exist: {DATA_PATH}")
        print("Please create data/company_docs directory and add documents")
        return
    
    try:
        # Step 1: Load and process documents
        print("\n📄 Step 1/3: Load and Process Documents")
        print("-" * 60)
        loader = DocumentLoader(DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
        chunks = loader.process_documents()
        
        if not chunks:
            print("❌ No documents found, please check data directory")
            return
        
        # Step 2: Create vector database
        print("\n🔄 Step 2/3: Generate Embeddings and Create Database")
        print("-" * 60)
        print("This may take a few minutes, please wait...")
        
        manager = VectorStoreManager(
            api_key=COHERE_API_KEY,
            persist_directory=VECTORDB_PATH,
            collection_name=COLLECTION_NAME
        )
        
        # If database exists, ask whether to overwrite
        if os.path.exists(VECTORDB_PATH):
            response = input("\n⚠️  Vector database already exists. Overwrite? (y/n): ")
            if response.lower() == 'y':
                manager.delete_collection()
                print("Old database deleted")
            else:
                print("Operation cancelled")
                return
        
        vectorstore = manager.create_vectorstore(chunks)
        
        # Step 3: Test database
        print("\n✅ Step 3/3: Test Vector Database")
        print("-" * 60)
        
        # Test queries
        test_queries = [
            "What is Digia",
            "What services are provided",
            "How to contact"
        ]
        
        print("\nRunning test queries:")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = manager.search(query, k=2)
            
            if results:
                print(f"Found {len(results)} relevant results:")
                for i, doc in enumerate(results, 1):
                    print(f"  Result {i}: {doc.page_content[:100]}...")
            else:
                print("  No relevant results found")
        
        print("\n" + "=" * 60)
        print("✅ Vector Database Build Complete!")
        print("=" * 60)
        print(f"📊 Statistics:")
        print(f"  - Number of chunks: {len(chunks)}")
        print(f"  - Storage location: {VECTORDB_PATH}")
        print(f"  - Collection name: {COLLECTION_NAME}")
        print("\n💡 Tip: You can now run the chat application")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()