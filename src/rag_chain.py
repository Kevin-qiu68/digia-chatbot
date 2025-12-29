import cohere
from typing import List, Dict, Optional
from langchain_core.documents import Document
from src.vectorstore import VectorStoreManager


class RAGChain:
    """RAG (Retrieval-Augmented Generation) Chain"""
    
    def __init__(
        self, 
        vectorstore_manager: VectorStoreManager,
        cohere_api_key: str,
        model: str = "command-r-plus",
        top_k_retrieval: int = 20,
        top_k_rerank: int = 3,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        self.vectorstore_manager = vectorstore_manager
        self.cohere_client = cohere.Client(cohere_api_key)
        self.model = model
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Load vectorstore
        if self.vectorstore_manager.vectorstore is None:
            self.vectorstore_manager.load_vectorstore()
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents using semantic search"""
        print(f"\n🔍 Retrieving documents for query: '{query}'")
        
        # Semantic search
        results = self.vectorstore_manager.search(query, k=self.top_k_retrieval)
        
        print(f"✓ Retrieved {len(results)} documents")
        return results
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Dict]:
        """Rerank documents using Cohere Rerank API"""
        print(f"\n🎯 Reranking top {self.top_k_rerank} documents...")
        
        # Prepare documents for reranking
        docs_text = [doc.page_content for doc in documents]
        
        # Call Cohere Rerank API
        rerank_response = self.cohere_client.rerank(
            query=query,
            documents=docs_text,
            top_n=self.top_k_rerank,
            model="rerank-multilingual-v3.0"
        )
        
        # Combine reranked results with original metadata
        reranked_docs = []
        for result in rerank_response.results:
            original_doc = documents[result.index]
            reranked_docs.append({
                "content": original_doc.page_content,
                "metadata": original_doc.metadata,
                "relevance_score": result.relevance_score
            })
        
        print(f"✓ Reranked to top {len(reranked_docs)} documents")
        return reranked_docs
    
    def build_context(self, reranked_docs: List[Dict]) -> str:
        """Build context string from reranked documents"""
        context_parts = []
        
        for i, doc in enumerate(reranked_docs, 1):
            source = doc["metadata"].get("source", "Unknown")
            content = doc["content"]
            score = doc["relevance_score"]
            
            context_parts.append(
                f"[Document {i}] (Source: {source}, Relevance: {score:.3f})\n{content}"
            )
        
        context = "\n\n".join(context_parts)
        return context
    
    def build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM"""
        prompt = f"""You are an AI assistant for Digia company. Your role is to answer customer questions based on the provided context.

Instructions:
- Answer questions accurately based on the context provided
- If the information is not in the context, say "I don't have that information in my knowledge base"
- Be concise and professional
- Respond in the same language as the question
- If asked about services, products, or company information, provide specific details from the context

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_response(self, prompt: str, chat_history: Optional[List[Dict]] = None) -> str:
        """Generate response using Cohere LLM"""
        print("\n💬 Generating response...")
        
        try:
            # Prepare chat history if provided
            if chat_history:
                response = self.cohere_client.chat(
                    message=prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    chat_history=chat_history
                )
            else:
                response = self.cohere_client.chat(
                    message=prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            
            answer = response.text
            print("✓ Response generated")
            return answer
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."
    
    def query(self, question: str, use_rerank: bool = True, chat_history: Optional[List[Dict]] = None) -> Dict:
        """Complete RAG query pipeline"""
        print("\n" + "=" * 60)
        print(f"Processing query: {question}")
        print("=" * 60)
        
        try:
            # Step 1: Retrieve documents
            retrieved_docs = self.retrieve_documents(question)
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "error": None
                }
            
            # Step 2: Rerank documents (optional)
            if use_rerank:
                reranked_docs = self.rerank_documents(question, retrieved_docs)
            else:
                # Use top-k retrieved documents without reranking
                reranked_docs = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": 1.0
                    }
                    for doc in retrieved_docs[:self.top_k_rerank]
                ]
            
            # Step 3: Build context
            context = self.build_context(reranked_docs)
            
            # Step 4: Build prompt
            prompt = self.build_prompt(question, context)
            
            # Step 5: Generate response
            answer = self.generate_response(prompt, chat_history)
            
            # Prepare sources
            sources = [
                {
                    "source": doc["metadata"].get("source", "Unknown"),
                    "relevance_score": doc["relevance_score"],
                    "content_preview": doc["content"][:150] + "..."
                }
                for doc in reranked_docs
            ]
            
            print("\n" + "=" * 60)
            print("✅ Query completed successfully")
            print("=" * 60)
            
            return {
                "answer": answer,
                "sources": sources,
                "error": None
            }
            
        except Exception as e:
            print(f"\n❌ Error in RAG pipeline: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "answer": "I apologize, but I encountered an error processing your question.",
                "sources": [],
                "error": str(e)
            }


if __name__ == "__main__":
    # Test code
    from config import COHERE_API_KEY, VECTORDB_PATH, COLLECTION_NAME, COHERE_MODEL, TOP_K_RETRIEVAL, TOP_K_RERANK
    
    print("Initializing RAG Chain...")
    
    # Initialize vector store manager
    vectorstore_manager = VectorStoreManager(
        api_key=COHERE_API_KEY,
        persist_directory=VECTORDB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # Initialize RAG chain
    rag_chain = RAGChain(
        vectorstore_manager=vectorstore_manager,
        cohere_api_key=COHERE_API_KEY,
        model=COHERE_MODEL,
        top_k_retrieval=TOP_K_RETRIEVAL,
        top_k_rerank=TOP_K_RERANK
    )
    
    # Test queries
    test_queries = [
        "What is Digia?",
        "What services does Digia provide?",
        "How can I contact Digia?"
    ]
    
    print("\n" + "=" * 60)
    print("Testing RAG Chain")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n\nQuery: {query}")
        print("-" * 60)
        
        result = rag_chain.query(query, use_rerank=True)
        
        print(f"\nAnswer:\n{result['answer']}")
        
        print(f"\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['source']} (Score: {source['relevance_score']:.3f})")
            print(f"     Preview: {source['content_preview']}")
        
        print("\n" + "=" * 60)