from datalab_sdk import DatalabClient
import os

# Global client instance (lazy loaded)
_client = None


def get_client():
    """
    Get or create the DatalabClient instance.
    
    Returns:
        DatalabClient: The client instance
    """
    global _client
    if _client is None:
        api_key = os.getenv("DATALAB_API_KEY")
        _client = DatalabClient(api_key=api_key)
    return _client


def run_ocr_inference(image_path: str) -> str:
    """
    Run OCR inference on an image using Chandra OCR model via API client.
    
    Args:
        image_path: Path to the input image file
    
    Returns:
        str: The extracted text from the image in markdown format
    """
    client = get_client()
    
    # Run OCR inference via API
    result = client.ocr(image_path)
    
    # Get text result
    text = result.get_text()
    
    return text