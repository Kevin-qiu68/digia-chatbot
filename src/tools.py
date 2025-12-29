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
    
    def run(self, query: str) -> str:
        """Execute knowledge base search"""
        print(f"🔧 Using tool: {self.name}")
        print(f"   Query: {query}")
        
        result = self.rag_chain.query(query, use_rerank=True)
        
        if result['error']:
            return f"Error searching knowledge base: {result['error']}"
        
        # Format response with sources
        response = result['answer']
        
        if result['sources']:
            response += "\n\nSources:"
            for i, source in enumerate(result['sources'], 1):
                response += f"\n{i}. {source['source']}"
        
        return response


class CalculatorTool:
    """Tool for performing calculations"""
    
    name = "calculator"
    description = """Perform mathematical calculations. 
    Use this when the user asks to calculate numbers, percentages, or other mathematical operations.
    Input should be a mathematical expression like '100 * 1.2' or '(500 + 300) / 2'."""
    
    def run(self, expression: str) -> str:
        """Execute calculation"""
        print(f"🔧 Using tool: {self.name}")
        print(f"   Expression: {expression}")
        
        try:
            # Safely evaluate mathematical expression
            # Only allow basic math operations
            allowed_chars = set("0123456789+-*/.()\n ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /, parentheses) are allowed."
            
            result = eval(expression)
            return f"Result: {result}"
        
        except Exception as e:
            return f"Error calculating: {str(e)}"


class CurrentTimeTool:
    """Tool for getting current time information"""
    
    name = "current_time"
    description = """Get the current date and time.
    Use this when the user asks about the current time, date, day of week, or year."""
    
    def run(self, query: str = "") -> str:
        """Get current time"""
        print(f"🔧 Using tool: {self.name}")
        
        now = datetime.now()
        
        return f"""Current date and time information:
- Date: {now.strftime('%Y-%m-%d')}
- Time: {now.strftime('%H:%M:%S')}
- Day of week: {now.strftime('%A')}
- Full: {now.strftime('%B %d, %Y at %I:%M %p')}"""


class ContactInfoTool:
    """Tool for getting contact information"""
    
    name = "get_contact_info"
    description = """Get Digia's contact information including email, phone, and address.
    Use this when the user specifically asks how to contact Digia or needs contact details."""
    
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
    
    def run(self, query: str = "contact information") -> str:
        """Get contact information"""
        print(f"🔧 Using tool: {self.name}")
        
        # First try to get from knowledge base
        result = self.rag_chain.query(f"Digia contact information: {query}", use_rerank=True)
        
        if result['answer'] and not result['error']:
            return result['answer']
        
        # Fallback generic message
        return """For contact information, please:
- Visit our website
- Check the contact section in our company documentation
- Or search for 'contact' or 'how to reach us' in the knowledge base"""


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
        },
        {
            "name": "get_contact_info",
            "description": """Get Digia's contact information including email, phone, and address.
            Use this when the user specifically asks how to contact Digia or needs contact details.""",
            "parameter_definitions": {
                "query": {
                    "description": "Optional specific contact information needed (e.g., 'email', 'phone', 'address')",
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