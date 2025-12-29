import os
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentLoader:
    """Load and process documents"""
    
    def __init__(self, data_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
    
    def load_documents(self) -> List[Document]:
        """Load all documents from the data directory"""
        documents = []
        data_dir = Path(self.data_path)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        # Recursively traverse all text files
        for file_path in data_dir.rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Get relative path as metadata
                relative_path = file_path.relative_to(data_dir)
                
                # Create Document object
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(relative_path),
                        "filename": file_path.name,
                        "category": relative_path.parts[0] if len(relative_path.parts) > 1 else "general"
                    }
                )
                documents.append(doc)
                print(f"✓ Loaded: {relative_path}")
                
            except Exception as e:
                print(f"✗ Failed to load {file_path}: {e}")
        
        print(f"\nTotal documents loaded: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"Documents split into {len(chunks)} chunks")
        return chunks
    
    def process_documents(self) -> List[Document]:
        """Complete document processing pipeline"""
        print("=" * 50)
        print("Starting document loading and processing...")
        print("=" * 50)
        
        # Load documents
        documents = self.load_documents()
        
        if not documents:
            raise ValueError("No documents found! Please check the data directory")
        
        # Split documents
        chunks = self.split_documents(documents)
        
        print("=" * 50)
        print(f"Processing complete! Total chunks: {len(chunks)}")
        print("=" * 50)
        
        return chunks


if __name__ == "__main__":
    # Test code
    from config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP
    
    loader = DocumentLoader(DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = loader.process_documents()
    
    # Display first chunk as example
    if chunks:
        print("\nExample chunk:")
        print("-" * 50)
        print(f"Content: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
        print("-" * 50)