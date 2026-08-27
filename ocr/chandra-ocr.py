from PIL import Image
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem

# Global manager instance (lazy loaded)
_manager = None


def get_manager():
    """
    Get or create the InferenceManager instance.
    
    Returns:
        InferenceManager: The manager instance
    """
    global _manager
    if _manager is None:
        _manager = InferenceManager(method="hf")
    return _manager


def run_ocr_inference(image_path: str, prompt_type: str = "ocr_layout") -> str:
    """
    Run OCR inference on an image using Chandra OCR model.
    
    Args:
        image_path: Path to the input image file
        prompt_type: Type of prompt to use (default: "ocr_layout")
    
    Returns:
        str: The extracted text from the image in markdown format
    """
    manager = get_manager()
    
    # Open image from file path
    pil_image = Image.open(image_path)
    
    # Create batch input
    batch = [
        BatchInputItem(
            image=pil_image,
            prompt_type=prompt_type
        )
    ]
    
    # Run inference
    result = manager.generate(batch)[0]
    
    return result.markdown