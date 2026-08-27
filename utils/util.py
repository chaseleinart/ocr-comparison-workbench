import os
import base64
from urllib.parse import urlparse

# -------------------------
# URL Validation
# -------------------------
def is_url(path_or_url: str) -> bool:
    """Check if the input is a valid URL."""
    try:
        result = urlparse(path_or_url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


# -------------------------
# Image Encoding
# -------------------------
def encode_image_to_base64(image_path: str) -> str:
    """Reads an image file from disk and encodes it into a Base64 data URI string."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    
    if not os.path.isfile(image_path):
        raise ValueError(f"Path is not a file: {image_path}")

    # Determine the image type (e.g., png, jpeg) from the file extension
    mime_type = "image/png" # Default
    if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
        mime_type = "image/jpeg"
    elif image_path.lower().endswith(".gif"):
        mime_type = "image/gif"
    
    # Read the image file in binary mode
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    # Encode the bytes into a Base64 string
    base64_str = base64.b64encode(image_bytes).decode("utf-8")

    # Format as a data URI
    data_uri = f"data:{mime_type};base64,{base64_str}"
    return data_uri