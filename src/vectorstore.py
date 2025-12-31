import cohere
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional
import os

class VectorStoreManager:
    """Vector database manager"""
    
    def __init__(self, api_key: str, persist_directory: str, collection_name: str):
        self.api_key = api_key
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize Cohere embeddings
        self.embeddings = CohereEmbeddings(
            cohere_api_key=api_key,
            model="embed-multilingual-v3.0"  # Supports multiple languages
        )
        
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create vector database"""
        print("\nCreating vector database...")
        print(f"Number of documents: {len(documents)}")
        print(f"Storage path: {self.persist_directory}")
        
        # Create vector database
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        print("✓ Vector database created successfully!")
        return self.vectorstore
    
    def load_vectorstore(self) -> Optional[Chroma]:
        """Load existing vector database"""
        if not os.path.exists(self.persist_directory):
            print(f"Vector database does not exist: {self.persist_directory}")
            return None
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print(f"✓ Loaded vector database with {self.vectorstore._collection.count()} documents")
            return self.vectorstore
        except Exception as e:
            print(f"Failed to load vector database: {e}")
            return None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to existing database"""
        if self.vectorstore is None:
            raise ValueError("Please create or load vector database first")
        
        self.vectorstore.add_documents(documents)
        print(f"✓ Added {len(documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents"""
        if self.vectorstore is None:
            raise ValueError("Please create or load vector database first")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def search_with_score(self, query: str, k: int = 5):
        """Search and return similarity scores"""
        if self.vectorstore is None:
            raise ValueError("Please create or load vector database first")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def delete_collection(self):
        """Delete entire collection"""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            print("✓ Vector database deleted")


if __name__ == "__main__":
    # Test code
    from config import COHERE_API_KEY, VECTORDB_PATH, COLLECTION_NAME
    
    manager = VectorStoreManager(
        api_key=COHERE_API_KEY,
        persist_directory=VECTORDB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # Try to load existing database
    vectorstore = manager.load_vectorstore()
    
    if vectorstore:
        # Test search
        test_query = "What services does Digia provide"
        print(f"\nTest query: {test_query}")
        results = manager.search(test_query, k=3)
        
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Content: {doc.page_content[:150]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    else:
        print("Vector database does not exist, please run build_vectordb.py first")