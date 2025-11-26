import base64
import os
from typing import Dict, List, Optional, Union

def create_input_file_from_path(file_path: str) -> Dict:
    """
    Create an input_file o input_image content item from a file path.
    Supports images and PDFs.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict with input_file content for Responses API
    """
    file_path_lower = file_path.lower()
    filename = os.path.basename(file_path)
    
    # Read file and encode as base64
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    base64_string = base64.b64encode(file_data).decode('utf-8')
    input_type = 'input_file'

    # Determine MIME type
    if file_path_lower.endswith('.pdf'):
        mime_type = 'application/pdf'
        input_type = 'input_file'
    elif file_path_lower.endswith(('.jpg', '.jpeg')):
        mime_type = 'image/jpeg'
        input_type = 'input_image'
    elif file_path_lower.endswith('.png'):
        mime_type = 'image/png'
        input_type = 'input_image'
    elif file_path_lower.endswith('.gif'):
        mime_type = 'image/gif'
        input_type = 'input_image'
    elif file_path_lower.endswith('.webp'):
        mime_type = 'image/webp'
        input_type = 'input_image'
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    return {
        "type": input_type,
        "filename": filename,
        "file_data": f"data:{mime_type};base64,{base64_string}"
    }


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