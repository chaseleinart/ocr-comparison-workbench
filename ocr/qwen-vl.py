import ollama


def run_ocr_inference(image_path: str) -> str:
    """
    Run OCR inference on an image using Qwen3-VL model via Ollama.

    Args:
        image_path: Path to the input image file

    Returns:
        str: The extracted text from the image
    """
    response = ollama.chat(
        model="qwen3-vl:2b",
        messages=[
            {
                "role": "user",
                "content": "Read all the text in the image.",
                "images": [image_path],
            }
        ],
    )

    return response.message.content