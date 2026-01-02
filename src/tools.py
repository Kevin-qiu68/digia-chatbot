"""
Agent tools for different functionalities
"""

from typing import List, Dict, Any
import json
from datetime import datetime


class KnowledgeBaseTool:
    """Tool for querying the knowledge base"""
    
    name = "knowledge_base_search"
    description = """Search the Digia company knowledge base for information about services, 
    products, company information, and frequently asked questions. 
    Use this when the user asks about Digia's business, offerings, or company details."""
    
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
    
    def run(self, query: str) -> dict:
        """Execute knowledge base search - returns context only, not final answer"""
        print(f"🔧 Using tool: {self.name}")
        print(f"   Query: {query}")
        
        # Only retrieve and rerank documents, don't generate answer yet
        retrieved_docs = self.rag_chain.retrieve_documents(query)
        
        if not retrieved_docs:
            return {
                    "context": "No relevant information found in the knowledge base.",
                    "sources": []
            }
        
        # Rerank documents
        reranked_docs = self.rag_chain.rerank_documents(query, retrieved_docs)
        
        # Return formatted context for the agent to use
        context_parts = []
        sources = []
        for i, doc in enumerate(reranked_docs, 1):
            source = doc["metadata"].get("source", "Unknown")
            content = doc["content"]
            score = doc["relevance_score"]
            
            context_parts.append(
                f"[Source {i}: {source}] (Relevance: {score:.2f})\n{content}"
            )

            sources.append({
                "source": source, 
                "relevance_score": score, 
                "content_preview": content[:150] + "..."
            })
        
        context = "\n\n".join(context_parts)
        
        return {
            "context": f"Here is the relevant information from the knowledge base:\n\n{context}",
            "sources": sources
        }



class CalculatorTool:
    """Tool for performing calculations"""
    
    name = "calculator"
    description = """Perform mathematical calculations. 
    Use this when the user asks to calculate numbers, percentages, or other mathematical operations.
    Input should be a mathematical expression like '100 * 1.2' or '(500 + 300) / 2'."""
    
    def run(self, expression: str) -> dict:
        """Execute calculation"""
        print(f"🔧 Using tool: {self.name}")
        print(f"   Expression: {expression}")
        
        try:
            # Safely evaluate mathematical expression
            # Only allow basic math operations
            allowed_chars = set("0123456789+-*/.()\n ")
            if not all(c in allowed_chars for c in expression):
                return {
                    "context": "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /, parentheses) are allowed.",
                    "sources": None
                }
            
            result = eval(expression)
            return {
                "context": f"Result: {result}",
                "sources": None 
            }
        
        except Exception as e:
            return {
                "context": f"Error calculating expression: {str(e)}",
                "sources": None
            }


class CurrentTimeTool:
    """Tool for getting current time information"""
    
    name = "current_time"
    description = """Get the current date and time.
    Use this when the user asks about the current time, date, day of week, or year."""
    
    def run(self, query: str = "") -> dict:
        """Get current time"""
        print(f"🔧 Using tool: {self.name}")
        
        now = datetime.now()
        
        return {
            "context": f"Current date and time information:\n- Date: {now.strftime('%Y-%m-%d')}\n- Time: {now.strftime('%H:%M:%S')}\n- Day of week: {now.strftime('%A')}\n- Full: {now.strftime('%B %d, %Y at %I:%M %p')}",
            "sources": None
        }




def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get tool definitions in the format required by Cohere"""
    
    tools = [
        {
            "name": "knowledge_base_search",
            "description": """Search the Digia company knowledge base for information about services, 
            products, company information, and frequently asked questions. 
            Use this when the user asks about Digia's business, offerings, or company details.""",
            "parameter_definitions": {
                "query": {
                    "description": "The search query to find relevant information in the knowledge base",
                    "type": "str",
                    "required": True
                }
            }
        },
        {
            "name": "calculator",
            "description": """Perform mathematical calculations. 
            Use this when the user asks to calculate numbers, percentages, or other mathematical operations.""",
            "parameter_definitions": {
                "expression": {
                    "description": "Mathematical expression to evaluate, e.g., '100 * 1.2' or '(500 + 300) / 2'",
                    "type": "str",
                    "required": True
                }
            }
        },
        {
            "name": "current_time",
            "description": """Get the current date and time.
            Use this when the user asks about the current time, date, day of week, or year.""",
            "parameter_definitions": {
                "query": {
                    "description": "Optional query about what time information is needed",
                    "type": "str",
                    "required": False
                }
            }
        }
    ]
    
    return tools


if __name__ == "__main__":
    # Test tools
    print("Testing Tools...")
    
    print("\n1. Calculator Tool:")
    calc = CalculatorTool()
    print(calc.run("100 + 50 * 2"))
    
    print("\n2. Current Time Tool:")
    time_tool = CurrentTimeTool()
    print(time_tool.run())
    
    print("\n3. Tool Definitions:")
    tools = get_tool_definitions()
    for tool in tools:
        print(f"\n- {tool['name']}: {tool['description'][:100]}...")