from typing import Dict

def create_input_text(text: str) -> Dict:
    """
    Create an input_text content item.
    
    Args:
        text: Text content
        
    Returns:
        Dict with input_text content for Responses API
    """
    return {
        "type": "input_text",
        "text": text
    }