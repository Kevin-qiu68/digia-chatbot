"""
Digia AI Assistant - Streamlit Web Application
"""

import streamlit as st
import sys
import os
from datetime import datetime

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


# Page configuration
st.set_page_config(
    page_title="Digia AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .tool-info {
        font-size: 0.85rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_agent():
    """Initialize and cache the agent system"""
    try:
        # Check API key
        if not COHERE_API_KEY:
            st.error("❌ COHERE_API_KEY not found in .env file")
            st.stop()
        
        # Check vector database
        if not os.path.exists(VECTORDB_PATH):
            st.error(f"❌ Vector database not found at {VECTORDB_PATH}")
            st.info("Please run build_vectordb.py first to create the knowledge base")
            st.stop()
        
        # Initialize components
        with st.spinner("Loading knowledge base..."):
            vectorstore_manager = VectorStoreManager(
                api_key=COHERE_API_KEY,
                persist_directory=VECTORDB_PATH,
                collection_name=COLLECTION_NAME
            )
        
        with st.spinner("Initializing RAG system..."):
            rag_chain = RAGChain(
                vectorstore_manager=vectorstore_manager,
                cohere_api_key=COHERE_API_KEY,
                model=COHERE_MODEL
            )
        
        with st.spinner("Starting AI agent..."):
            agent = DigiaAgent(
                cohere_api_key=COHERE_API_KEY,
                rag_chain=rag_chain,
                model=COHERE_MODEL
            )
        
        return agent, rag_chain
    
    except Exception as e:
        st.error(f"❌ Error initializing agent: {e}")
        st.stop()


def display_message(role, content, tool_info=None, sources=None):
    """Display a chat message with styling"""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    
    st.markdown(f"""
        <div class="chat-message {css_class}">
            <div><strong>{icon} {role.title()}</strong></div>
            <div style="margin-top: 0.5rem;">{content}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Display tool info if available
    if tool_info:
        st.markdown(f'<div class="tool-info">🔧 {tool_info}</div>', unsafe_allow_html=True)
    
    # Display sources if available
    if sources:
        with st.expander("📚 View Sources", expanded=False):
            for i, source in enumerate(sources, 1):
                st.markdown(f"""
                    <div class="source-box">
                        <strong>{i}. {source.get('source', 'Unknown')}</strong><br>
                        Relevance: {source.get('relevance_score', 0):.3f}<br>
                        Preview: {source.get('content_preview', 'N/A')}
                    </div>
                """, unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🤖 Digia AI Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize agent
    agent, rag_chain = initialize_agent()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Select Mode:",
            ["Agent Mode (Recommended)", "RAG Only Mode"],
            help="Agent mode can use tools like calculator and time. RAG mode only searches knowledge base."
        )
        
        use_agent = mode == "Agent Mode (Recommended)"
        
        st.markdown("---")
        
        # Display capabilities
        st.subheader("🎯 Capabilities")
        if use_agent:
            st.markdown("""
            - 📚 Search knowledge base
            - 🔢 Perform calculations
            - 🕐 Get current time
            - 📞 Provide contact info
            - 💬 Multi-turn conversations
            """)
        else:
            st.markdown("""
            - 📚 Search knowledge base
            - 💬 Answer questions about Digia
            """)
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📊 Session Stats")
        if "message_count" not in st.session_state:
            st.session_state.message_count = 0
        st.metric("Messages Sent", st.session_state.message_count)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.message_count = 0
            st.rerun()
        
        st.markdown("---")
        
        # Sample questions
        st.subheader("💡 Try asking:")
        sample_questions = [
            "What is Digia?",
            "What services do you provide?",
            "Calculate 150 * 3",
            "What's the current date?",
            "How can I contact you?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.pending_question = question
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Hello! I'm the Digia AI Assistant. How can I help you today?")
        
        for message in st.session_state.messages:
            display_message(
                message["role"],
                message["content"],
                message.get("tool_info"),
                message.get("sources")
            )
    
    # Chat input
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.chat_input("Type your message here...")
    
    # Handle pending question from sidebar
    if "pending_question" in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question
    
    # Process user input
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        st.session_state.message_count += 1
        
        # Display user message
        with chat_container:
            display_message("user", user_input)
        
        # Get response
        with st.spinner("Thinking..."):
            try:
                if use_agent:
                    # Use agent mode
                    result = agent.run(user_input, chat_history=st.session_state.chat_history)
                    
                    response_text = result["answer"]
                    tool_info = None
                    sources = result.get("sources", [])  # Get sources from agent if available
                    
                    # Add tool information
                    if result["tool_calls_made"]:
                        tool_info = f"Used tools in {result['iterations']} iteration(s)"
                    
                    # Update chat history
                    st.session_state.chat_history = result.get("chat_history", [])
                
                else:
                    # Use RAG only mode
                    result = rag_chain.query(user_input, use_rerank=True)
                    
                    response_text = result["answer"]
                    tool_info = "RAG retrieval with reranking"
                    sources = result.get("sources", [])
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "tool_info": tool_info,
                    "sources": sources
                })
                
                # Display assistant message
                with chat_container:
                    display_message("assistant", response_text, tool_info, sources)
            
            except Exception as e:
                error_message = f"I apologize, but I encountered an error: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                
                with chat_container:
                    display_message("assistant", error_message)
        
        # Rerun to update the display
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.85rem;">
            Powered by Cohere AI | Built with Streamlit | RAG + Agent Architecture
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()