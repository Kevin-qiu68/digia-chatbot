# 🤖 Digia AI Assistant

An intelligent customer service chatbot powered by RAG (Retrieval-Augmented Generation) and Agent architecture, built with Cohere AI.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [What I Learned](#what-i-learned)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project is an AI-powered chatbot designed for Digia company to answer customer questions about services, products, and company information. It combines two powerful AI techniques:

- **RAG (Retrieval-Augmented Generation)**: Searches a vector database of company documents to provide accurate, source-backed answers
- **Agent System**: Uses tools to perform calculations, get current time, and orchestrate complex multi-step tasks

## ✨ Features

### Core Capabilities
- 📚 **Knowledge Base Search**: Semantic search through company documents with source attribution
- 🔢 **Calculator**: Perform mathematical calculations
- 🕐 **Time Information**: Get current date and time
- 📞 **Contact Information**: Retrieve company contact details
- 💬 **Multi-turn Conversations**: Maintains context across conversation history

### Technical Features
- **Semantic Search**: Uses Cohere embeddings for accurate document retrieval
- **Reranking**: Cohere Rerank API improves retrieval precision
- **Tool Calling**: Agent autonomously decides which tools to use
- **Dual Modes**: Switch between Agent mode (all tools) and RAG-only mode
- **Source Tracking**: Shows which documents were used to generate answers
- **Responsive UI**: Clean, modern Streamlit interface

## 🏗️ Architecture

```
User Query
    ↓
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Agent System   │ ← Decides which tools to use
└────────┬────────┘
         ↓
    ┌────┴────┐
    ↓         ↓
┌─────────┐ ┌──────────────┐
│  Tools  │ │  RAG Chain   │
└─────────┘ └──────┬───────┘
                   ↓
            ┌──────────────┐
            │ Vector Store │ ← Chroma DB with Cohere embeddings
            └──────────────┘
                   ↓
            ┌──────────────┐
            │ Cohere LLM   │ ← Command-R-Plus model
            └──────────────┘
```

### Component Breakdown

1. **Data Loader**: Processes company documents and splits them into chunks
2. **Vector Store**: Stores document embeddings using Chroma DB
3. **RAG Chain**: 
   - Retrieves relevant documents via semantic search
   - Reranks results using Cohere Rerank API
   - Generates responses using retrieved context
4. **Agent System**:
   - Analyzes user queries
   - Selects appropriate tools
   - Orchestrates multi-step reasoning
5. **Tools**:
   - Knowledge Base Search
   - Calculator
   - Current Time
   - Contact Information

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- Cohere API key ([Get one here](https://dashboard.cohere.com/))

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/Kevin-qiu68/digia-chatbot.git
cd digia-chatbot
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root:
```env
COHERE_API_KEY=your_cohere_api_key_here
```

5. **Prepare company documents**

Place your company documents in `data/company_docs/`:
```
data/
└── company_docs/
    ├── about/
    │   └── company_intro.txt
    ├── services/
    │   └── service_list.txt
    └── products/
        └── product_info.txt
```

6. **Build vector database**
```bash
python build_vectordb.py
```

7. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

### Testing Components

**Test RAG System:**
```bash
python test_rag.py
```

**Test Agent System:**
```bash
python test_agent.py
```

### Using the Web Interface

1. **Start the app**: `streamlit run app.py`
2. **Select mode**: Choose between Agent Mode or RAG-only Mode in the sidebar
3. **Ask questions**: Type your question in the chat input
4. **View sources**: Click "View Sources" to see document references
5. **Try samples**: Use quick question buttons in the sidebar

### Example Queries

**Knowledge Base Questions:**
- "What is Digia?"
- "What services does Digia provide?"
- "Tell me about Digia's products"

**Calculator:**
- "Calculate 150 * 3"
- "What is (500 + 300) / 2?"

**Time Information:**
- "What is the current date?"
- "What day of the week is it?"

**Multi-tool Queries:**
- "Tell me about Digia's services and calculate 100 * 5"

## 📁 Project Structure

```
digia-chatbot/
├── venv/                      # Virtual environment (not in git)
├── data/
│   └── company_docs/          # Company documents for RAG
├── vectordb/                  # Chroma vector database (not in git)
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and constants
│   ├── data_loader.py         # Document loading and processing
│   ├── vectorstore.py         # Vector database management
│   ├── rag_chain.py           # RAG retrieval and generation
│   ├── agent.py               # Agent system with tool calling
│   └── tools.py               # Tool definitions
├── app.py                     # Streamlit web application
├── build_vectordb.py          # Script to build vector database
├── test_rag.py                # RAG system tests
├── test_agent.py              # Agent system tests
├── .env                       # Environment variables (not in git)
├── .gitignore
├── requirements.txt           # Python dependencies
└── README.md
```

## 🎓 What I Learned

### Technical Skills

**1. RAG (Retrieval-Augmented Generation)**
- Implemented end-to-end RAG pipeline from scratch
- Learned how to chunk documents effectively for optimal retrieval
- Understanding the importance of chunk size and overlap in semantic search
- Experienced the significant improvement that reranking provides over basic vector search
- Learned to balance retrieval quantity vs. quality (top-k tuning)

**2. Vector Databases**
- Hands-on experience with Chroma DB for storing and querying embeddings
- Understanding of semantic search vs. keyword search
- Learned about embedding models and their impact on search quality
- Practical experience with persistence and database management

**3. Agent Systems**
- Built a ReAct (Reasoning + Acting) agent from scratch
- Implemented tool calling and learned how LLMs can autonomously select tools
- Understanding multi-step reasoning and task decomposition
- Learned to handle tool execution errors and edge cases
- Experienced the power of combining different capabilities (search, calculation, etc.)

**4. Prompt Engineering**
- Crafted effective system prompts for different use cases
- Learned the importance of clear instructions and examples
- Understanding how to constrain model behavior while maintaining flexibility
- Experienced the difference between prompts for RAG vs. general chat

**5. LLM APIs (Cohere)**
- Practical experience with commercial LLM APIs
- Understanding API rate limits, token usage, and cost management
- Learned about different model capabilities (Command-R-Plus, Rerank)
- Experience with chat history management and conversation state

**6. Full-Stack Development**
- Built a complete application from data processing to user interface
- Learned Streamlit for rapid web app development
- Understanding state management in web applications
- Experience with async operations and loading states

### Software Engineering Practices

**1. Modular Architecture**
- Designed clean separation of concerns (data, retrieval, generation, UI)
- Learned to create reusable components and utilities
- Understanding dependency injection and loose coupling

**2. Error Handling**
- Implemented comprehensive try-catch blocks throughout the system
- Learned to provide meaningful error messages to users
- Understanding graceful degradation when components fail

**3. Testing**
- Created separate test scripts for different components
- Learned the importance of testing in isolation vs. integration
- Understanding edge cases and how to test for them

**4. Configuration Management**
- Centralized configuration using config files
- Learned to use environment variables for sensitive data
- Understanding the importance of making systems configurable

**5. Version Control**
- Practiced meaningful commit messages
- Learned proper .gitignore patterns for Python projects
- Understanding what should and shouldn't be version controlled

### Domain Knowledge

**1. Information Retrieval**
- Understanding precision vs. recall tradeoffs
- Learned about relevance scoring and ranking
- Experience with source attribution and citations

**2. Natural Language Processing**
- Understanding embeddings and semantic similarity
- Learned about context windows and token limits
- Experience with multilingual models

**3. User Experience**
- Learned to design conversational interfaces
- Understanding the importance of feedback and loading states
- Experience with progressive disclosure (expandable sources)

### Challenges Overcome

**1. Chat History Management**
- Problem: Cohere's chat history format didn't match my initial implementation
- Solution: Learned to work with API-specific data structures and format conversations properly
- Lesson: Always read API documentation carefully and handle both dict and object types

**2. Vector Database Persistence**
- Problem: Database not persisting correctly between sessions
- Solution: Proper directory management and understanding Chroma's persistence model
- Lesson: Storage and retrieval patterns require careful attention to file paths and permissions

**3. Tool Execution Context**
- Problem: Tools needed access to other components (like RAG chain)
- Solution: Dependency injection pattern in tool initialization
- Lesson: Design patterns matter even in small projects

**4. Source Tracking in Agent Mode**
- Problem: Agent mode wasn't returning document sources like RAG mode
- Solution: Modified agent to capture and pass through sources from knowledge base tool
- Lesson: Maintain consistent data structures across different execution paths

### Key Takeaways

1. **RAG is powerful but requires tuning**: Chunk size, retrieval count, and reranking all significantly impact quality
2. **Agents add flexibility**: Tool calling enables much more than simple Q&A
3. **User feedback is crucial**: Source attribution and tool usage visibility build trust
4. **Start simple, iterate**: Built basic RAG first, then added agents and UI
5. **Testing matters**: Separate test scripts caught issues before UI integration
6. **Documentation is for future you**: Clear code comments and README save time later

### Skills Applicable to Job Market

- **Production RAG Systems**: Can implement and deploy retrieval-augmented generation
- **LLM Integration**: Experience with commercial APIs and managing costs
- **Agent Development**: Understanding of autonomous AI systems with tool use
- **Full-Stack AI Apps**: Can build complete applications from data to deployment
- **Python Best Practices**: Clean code, modular design, proper project structure
- **Web Development**: Streamlit for rapid prototyping and deployment

## 🛠️ Technologies Used

- **Python 3.9+**: Core programming language
- **Cohere AI**: LLM API for embeddings, reranking, and generation
  - Command-R-Plus: Main language model
  - Embed-Multilingual: Document embeddings
  - Rerank-Multilingual: Result reranking
- **LangChain**: Framework for LLM applications
- **Chroma DB**: Vector database for semantic search
- **Streamlit**: Web application framework
- **Python-dotenv**: Environment variable management

## 🔮 Future Improvements

### Short-term
- [ ] Add user authentication and session management
- [ ] Implement conversation export (download chat history)
- [ ] Add feedback mechanism (thumbs up/down on responses)
- [ ] Support file upload for dynamic knowledge base updates
- [ ] Add more tools (web search, email sending, etc.)

### Long-term
- [ ] Multi-language support in UI
- [ ] Voice input/output capabilities
- [ ] Fine-tune custom embedding model on company data
- [ ] Implement caching to reduce API costs
- [ ] A/B testing framework for prompt optimization
- [ ] Analytics dashboard for usage statistics
- [ ] Deploy to cloud (Streamlit Cloud, AWS, Azure)
- [ ] Mobile app version

## 📝 Requirements

```txt
cohere>=5.0.0
langchain>=0.1.0
langchain-cohere>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.0
streamlit>=1.30.0
python-dotenv>=1.0.0
```

## 🤝 Contributing

This is a learning project, but suggestions and improvements are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share feedback on architecture or implementation

## 🙏 Acknowledgments

- Cohere AI for providing excellent API documentation and models
- LangChain community for the framework and examples
- Streamlit for the rapid prototyping capabilities

## 📧 Contact

For questions or feedback about this project:
- Create an issue in the repository
- Connect with me on LinkedIn: www.linkedin.com/in/keqiu1

---

**Note**: This project was built as a learning exercise to understand RAG systems, agent architectures, and LLM application development. It demonstrates practical implementation of modern AI technologies in a real-world use case.