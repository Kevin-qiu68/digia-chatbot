"""
Test script for Agent system
Run this to test the agent's tool calling and reasoning capabilities
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import (
    COHERE_API_KEY,
    VECTORDB_PATH,
    COLLECTION_NAME,
    COHERE_MODEL
)
from src.vectorstore import VectorStoreManager
from src.rag_chain import RAGChain
from src.agent import DigiaAgent


def test_knowledge_base_tool(agent: DigiaAgent):
    """Test knowledge base search tool"""
    print("\n" + "=" * 60)
    print("TEST 1: Knowledge Base Tool")
    print("=" * 60)
    
    queries = [
        "What is Digia?",
        "What services does Digia provide?",
        "Tell me about Digia's products"
    ]
    
    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print('─' * 60)
        
        result = agent.run(query)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"Tool calls made: {result['tool_calls_made']}")
        print(f"Iterations: {result['iterations']}")


def test_calculator_tool(agent: DigiaAgent):
    """Test calculator tool"""
    print("\n" + "=" * 60)
    print("TEST 2: Calculator Tool")
    print("=" * 60)
    
    queries = [
        "What is 150 multiplied by 3?",
        "Calculate (500 + 300) divided by 2",
        "What's 100 plus 200 times 3?"
    ]
    
    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print('─' * 60)
        
        result = agent.run(query)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"Tool calls made: {result['tool_calls_made']}")


def test_time_tool(agent: DigiaAgent):
    """Test current time tool"""
    print("\n" + "=" * 60)
    print("TEST 3: Current Time Tool")
    print("=" * 60)
    
    queries = [
        "What is the current date?",
        "What time is it now?",
        "What day of the week is it?"
    ]
    
    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print('─' * 60)
        
        result = agent.run(query)
        
        print(f"\nAnswer: {result['answer']}")


def test_contact_tool(agent: DigiaAgent):
    """Test contact information tool"""
    print("\n" + "=" * 60)
    print("TEST 4: Contact Information Tool")
    print("=" * 60)
    
    queries = [
        "How can I contact Digia?",
        "What is Digia's email address?",
        "Give me Digia's contact information"
    ]
    
    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print('─' * 60)
        
        result = agent.run(query)
        
        print(f"\nAnswer: {result['answer']}")


def test_multi_tool_queries(agent: DigiaAgent):
    """Test queries that require multiple tools"""
    print("\n" + "=" * 60)
    print("TEST 5: Multi-Tool Queries")
    print("=" * 60)
    
    queries = [
        "Tell me about Digia's services and calculate 100 times 5",
        "What is Digia and what's the current date?",
        "Search for Digia products and calculate 250 + 150"
    ]
    
    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print('─' * 60)
        
        result = agent.run(query)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"Tool calls made: {result['tool_calls_made']}")
        print(f"Iterations: {result['iterations']}")


def test_conversation_flow(agent: DigiaAgent):
    """Test multi-turn conversation"""
    print("\n" + "=" * 60)
    print("TEST 6: Multi-Turn Conversation")
    print("=" * 60)
    
    conversation = [
        "What services does Digia provide?",
        "Can you calculate the cost for 3 of those services at 500 each?",
        "What's the current date?"
    ]
    
    chat_history = []
    
    for i, query in enumerate(conversation, 1):
        print(f"\n{'─' * 60}")
        print(f"Turn {i}: {query}")
        print('─' * 60)
        
        result = agent.run(query, chat_history=chat_history)
        
        print(f"\nAnswer: {result['answer']}")
        
        # Update chat history
        chat_history = result['chat_history']


def test_edge_cases(agent: DigiaAgent):
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("TEST 7: Edge Cases")
    print("=" * 60)
    
    queries = [
        "Tell me about something Digia doesn't do",
        "Calculate xyz + abc",  # Invalid expression
        "What happened in the year 3000?"  # Future date
    ]
    
    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print('─' * 60)
        
        result = agent.run(query)
        
        print(f"\nAnswer: {result['answer']}")
        if result['error']:
            print(f"Error: {result['error']}")


def interactive_mode(agent: DigiaAgent):
    """Interactive testing mode"""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Chat with the agent (type 'quit' to exit)")
    print("The agent can:")
    print("  - Search the knowledge base")
    print("  - Perform calculations")
    print("  - Tell you the current time")
    print("  - Provide contact information")
    
    chat_history = []
    
    while True:
        query = input("\n\nYou: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive mode...")
            break
        
        if not query:
            continue
        
        print("\nAgent:", end=" ")
        result = agent.run(query, chat_history=chat_history)
        print(result['answer'])
        
        # Update chat history
        chat_history = result['chat_history']
        
        # Show metadata
        if result['tool_calls_made']:
            print(f"\n[Used tools in {result['iterations']} iteration(s)]")


def main():
    """Main test function"""
    print("\n" + "=" * 70)
    print("AGENT SYSTEM TEST SUITE")
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
        # Initialize components
        print("\nInitializing Agent System...")
        
        # Initialize RAG chain
        vectorstore_manager = VectorStoreManager(
            api_key=COHERE_API_KEY,
            persist_directory=VECTORDB_PATH,
            collection_name=COLLECTION_NAME
        )
        
        rag_chain = RAGChain(
            vectorstore_manager=vectorstore_manager,
            cohere_api_key=COHERE_API_KEY,
            model=COHERE_MODEL
        )
        
        # Initialize agent
        agent = DigiaAgent(
            cohere_api_key=COHERE_API_KEY,
            rag_chain=rag_chain,
            model=COHERE_MODEL
        )
        
        print("✓ Agent system initialized successfully")
        
        # Test menu
        print("\n" + "=" * 70)
        print("Select test mode:")
        print("  1. Run all automated tests")
        print("  2. Test knowledge base tool")
        print("  3. Test calculator tool")
        print("  4. Test time tool")
        print("  5. Test contact tool")
        print("  6. Test multi-tool queries")
        print("  7. Test conversation flow")
        print("  8. Test edge cases")
        print("  9. Interactive mode")
        print("=" * 70)
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            test_knowledge_base_tool(agent)
            test_calculator_tool(agent)
            test_time_tool(agent)
            test_contact_tool(agent)
            test_multi_tool_queries(agent)
            test_conversation_flow(agent)
            test_edge_cases(agent)
        elif choice == '2':
            test_knowledge_base_tool(agent)
        elif choice == '3':
            test_calculator_tool(agent)
        elif choice == '4':
            test_time_tool(agent)
        elif choice == '5':
            test_contact_tool(agent)
        elif choice == '6':
            test_multi_tool_queries(agent)
        elif choice == '7':
            test_conversation_flow(agent)
        elif choice == '8':
            test_edge_cases(agent)
        elif choice == '9':
            interactive_mode(agent)
        else:
            print("Invalid choice. Running all tests...")
            test_knowledge_base_tool(agent)
            test_calculator_tool(agent)
            test_multi_tool_queries(agent)
        
        print("\n" + "=" * 70)
        print("✅ Testing Complete!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()