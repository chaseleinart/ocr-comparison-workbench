import requests
from utils.util import is_url


def run_ocr_inference(image_path: str, url: str, return_layout_items: bool = False) -> str:
    """
    Run OCR inference on an image using Dots OCR API.
    
    Args:
        image_path: URL to an image
        url: URL of the OCR API endpoint
        return_layout_items: Whether to return layout items along with text (default: False)
    
    Returns:
        str: The extracted text from the image (final_text)
        If return_layout_items is True, returns a dict with 'final_text' and 'layout_items'
    """
    # Check if input is a URL - if so, pass it directly to the API
    if is_url(image_path):
        payload = {
            "image_path": image_path,
        }
    
        response = requests.post(url, json=payload)
        response.raise_for_status()
    
        data = response.json()
    
        result = {
            "final_text": data["final_text"],
            "layout_items": data["layout_items"],
        } if return_layout_items else data["final_text"]
        
        return result
    else:
        raise ValueError("Invalid image URL")