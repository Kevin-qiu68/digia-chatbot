"""
Agent system with tool calling and reasoning capabilities
"""

import cohere
from typing import List, Dict, Any, Optional
import json

from src.tools import (
    KnowledgeBaseTool,
    CalculatorTool,
    CurrentTimeTool,
    get_tool_definitions
)


class DigiaAgent:
    """Intelligent agent for Digia customer service"""
    
    def __init__(
        self,
        cohere_api_key: str,
        rag_chain,
        model: str = "command-a-03-2025",
        temperature: float = 0.3,
        max_iterations: int = 5
    ):
        self.cohere_client = cohere.Client(cohere_api_key)
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        
        # Initialize tools
        self.tools_map = {
            "knowledge_base_search": KnowledgeBaseTool(rag_chain),
            "calculator": CalculatorTool(),
            "current_time": CurrentTimeTool()
        }
        
        # Get tool definitions for Cohere
        self.tool_definitions = get_tool_definitions()
        
        # System preamble
        self.preamble = """You are an intelligent AI assistant for Digia company. Your role is to help customers by:

        1. Answering questions about Digia's services, products, and company information
        2. Performing calculations when needed
        3. Providing current date/time information
        4. Helping users find contact information

        Guidelines:
        - Be professional, friendly, and helpful
        - Use tools when appropriate to provide accurate information
        - If you need information from the knowledge base, use the knowledge_base_search tool
        - For calculations, use the calculator tool
        - For time-related queries, use the current_time tool
        - Always provide clear and concise answers
        - If you don't have information, be honest about it"""
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> dict:
        """Execute a tool with given parameters"""
        if tool_name not in self.tools_map:
            return f"Error: Tool '{tool_name}' not found"
        
        tool = self.tools_map[tool_name]
        
        try:
            # Extract parameters based on tool
            if tool_name == "knowledge_base_search":
                query = parameters.get("query", "")
                return tool.run(query)
            
            elif tool_name == "calculator":
                expression = parameters.get("expression", "")
                return tool.run(expression)
            
            elif tool_name == "current_time":
                query = parameters.get("query", "")
                return tool.run(query)
            
            elif tool_name == "get_contact_info":
                query = parameters.get("query", "")
                return tool.run(query)
            
            else:
                return {
                    "context": f"Error: Unknown tool execution pattern for '{tool_name}'",
                    "sources": None
                }
        
        except Exception as e:
            return {
                "context": f"Error executing tool '{tool_name}': {str(e)}",
                "sources": None
            }
    
    def format_chat_history(self, history: List) -> List[Dict]:
        """Format chat history for Cohere API"""
        formatted = []
        for entry in history:
            # Handle both dict and Cohere object types
            if isinstance(entry, dict):
                role = entry.get("role", "").upper()
                message = entry.get("message", "")
                
                # Convert role names to Cohere format
                if role in ["USER", "CHATBOT", "SYSTEM", "TOOL"]:
                    formatted.append(entry)
                elif role == "ASSISTANT":
                    formatted.append({
                        "role": "CHATBOT",
                        "message": message
                    })
                else:
                    # Default to USER if unknown
                    formatted.append({
                        "role": "USER",
                        "message": message
                    })
            else:
                # Handle Cohere message objects
                try:
                    # Cohere objects have role and message attributes
                    if hasattr(entry, 'role') and hasattr(entry, 'message'):
                        formatted.append({
                            "role": entry.role.upper() if isinstance(entry.role, str) else str(entry.role),
                            "message": entry.message
                        })
                    else:
                        # Skip if we can't parse it
                        continue
                except Exception:
                    # Skip problematic entries
                    continue
        
        return formatted
    
    def run(self, user_message: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """Run agent with tool calling"""
        print("\n" + "=" * 60)
        print(f"Agent Processing: {user_message}")
        print("=" * 60)
        
        # Handle chat history - only format if provided and not empty
        formatted_history = []
        if chat_history:
            formatted_history = self.format_chat_history(chat_history)
        
        # Track sources from knowledge base queries
        all_sources = []
        
        try:
            # First API call with user message
            print(f"\n🤖 Calling Cohere API with user message")
            
            # Only pass chat_history if we have formatted history
            api_kwargs = {
                "message": user_message,
                "model": self.model,
                "temperature": self.temperature,
                "tools": self.tool_definitions,
                "preamble": self.preamble
            }
            
            if formatted_history:
                api_kwargs["chat_history"] = formatted_history
            
            response = self.cohere_client.chat(**api_kwargs)
            
            iteration = 1
            
            # Handle tool calls in a loop
            while response.tool_calls and iteration < self.max_iterations:
                print(f"\n🔧 Agent wants to use {len(response.tool_calls)} tool(s)")
                
                # Execute each tool call
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call.name
                    parameters = tool_call.parameters
                    
                    print(f"\n   Tool: {tool_name}")
                    print(f"   Parameters: {parameters}")
                    
                    # Execute tool
                    tool_result = self.execute_tool(tool_name, parameters)
                    result = tool_result.get("context", "")
                    sources = tool_result.get("sources", None)
                    
                    # Track sources from knowledge base
                    if sources:
                        print(f"agent sources: {sources}")
                        all_sources.extend(sources)
                    
                    tool_results.append({
                        "call": tool_call,
                        "outputs": [{"output": result}]
                    })
                    
                    print(f"   Result: {result[:100]}...")
                
                # Continue conversation with tool results
                iteration += 1
                print(f"\n🤖 Agent Iteration {iteration}/{self.max_iterations}")
                
                response = self.cohere_client.chat(
                    message="",
                    model=self.model,
                    temperature=self.temperature,
                    chat_history=response.chat_history,
                    tools=self.tool_definitions,
                    tool_results=tool_results,
                    preamble=self.preamble
                )
            
            # Get final answer
            print(f"\n✅ Agent completed in {iteration} iteration(s)")
            
            # Return the chat_history from response for next turn
            return {
                "answer": response.text,
                "tool_calls_made": iteration > 1 or bool(response.tool_calls),
                "iterations": iteration,
                "chat_history": response.chat_history if hasattr(response, 'chat_history') else [],
                "sources": all_sources if all_sources else None,
                "error": None
            }
        
        except Exception as e:
            print(f"\n❌ Error in agent execution: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                "tool_calls_made": False,
                "iterations": 0,
                "chat_history": formatted_history,
                "sources": None,
                "error": str(e)
            }
        
        except Exception as e:
            print(f"\n❌ Error in agent execution: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                "tool_calls_made": False,
                "iterations": 0,
                "chat_history": formatted_history,
                "sources": None,
                "error": str(e)
            }
    
    def run_simple(self, user_message: str) -> str:
        """Simple run method that returns only the answer"""
        result = self.run(user_message)
        return result["answer"]


if __name__ == "__main__":
    # Test agent
    from config import COHERE_API_KEY, VECTORDB_PATH, COLLECTION_NAME, COHERE_MODEL
    from vectorstore import VectorStoreManager
    from rag_chain import RAGChain
    
    print("Initializing Agent System...")
    
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
    
    print("✓ Agent initialized\n")
    
    # Test queries
    test_queries = [
        "What is Digia?",
        "What is 150 multiplied by 3?",
        "What is the current date?",
        "Tell me about Digia's services and calculate 100 + 200"
    ]
    
    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print("=" * 70)
        
        result = agent.run(query)
        
        print(f"\n📝 Answer:")
        print(result['answer'])
        
        print(f"\n📊 Metadata:")
        print(f"   - Tool calls made: {result['tool_calls_made']}")
        print(f"   - Iterations: {result['iterations']}")
        print(f"   - Error: {result['error']}")
        
        print("\n" + "=" * 70)