import asyncio
from typing import Any, Union
from ocr import (
    load_model,
    deepseek_ocr_inference,
    granite_ocr_inference,
    qwen_ocr_inference,
    chandra_ocr_inference,
    dots_ocr_inference,
)


# OCR Models registry
AVAILABLE_OCR_MODELS = {
    "DeepSeek OCR": "deepseek_ocr",
    "Qwen3-VL": "qwen_vl",
    "Granite Docling": "granite_docling",
    "Chandra OCR": "chandra_ocr",
    # "Dots OCR": "dots_ocr",
}


def get_all_ocr_model_names():
    """Get all available OCR model names for dropdown selection."""
    try:
        return list(AVAILABLE_OCR_MODELS.keys())
    except Exception as e:
        print(f"Error getting OCR model names: {e}")
        return []


def validate_ocr_model_name(model_name: str) -> bool:
    """Validate if an OCR model name exists in available models."""
    return model_name in AVAILABLE_OCR_MODELS


async def run_ocr_inference(
    ocr_model_name: str, image_input: Union[str, bytes, Any], **kwargs
) -> str:
    """
    Unified OCR inference function that routes to the appropriate OCR model.

    Args:
        ocr_model_name: Name of the OCR model to use (e.g., "DeepSeek OCR", "Qwen3-VL", "Granite Docling")
        image_input: Image input - can be:
            - str: Path to image file
        **kwargs: Additional arguments specific to each OCR model

    Returns:
        str: Extracted text from the image

    Raises:
        ValueError: If OCR model name is not found
        Exception: If OCR inference fails
    """
    if not validate_ocr_model_name(ocr_model_name):
        raise ValueError(
            f"OCR model '{ocr_model_name}' not found in available OCR models"
        )

    try:
        # Route to appropriate OCR model
        ocr_model_key = AVAILABLE_OCR_MODELS[ocr_model_name]

        if ocr_model_key == "deepseek_ocr":
            # Load model if not already loaded (you may want to cache this)
            model, tokenizer = load_model(model_name="./deepseek_ocr")
            # Run inference
            result = await asyncio.to_thread(
                deepseek_ocr_inference,
                model=model,
                tokenizer=tokenizer,
                image_file=image_input,
                output_path="./result",
            )
            return result

        elif ocr_model_key == "granite_docling":
            result = await asyncio.to_thread(
                granite_ocr_inference,
                image_path=image_input,
            )
            return result

        elif ocr_model_key == "qwen_vl":
            result = await asyncio.to_thread(
                qwen_ocr_inference,
                image_path=image_input,
            )
            return result

        elif ocr_model_key == "chandra_ocr":
            result = await asyncio.to_thread(
                chandra_ocr_inference,
                image_path=image_input,
            )
            return result

        elif ocr_model_key == "dots_ocr":
            result = await asyncio.to_thread(
                dots_ocr_inference,
                image_path=image_input,
                url="<litserver-url>",
            )
            return result

        else:
            raise ValueError(f"Unsupported OCR model key: {ocr_model_key}")

    except Exception as e:
        print(f"Error running OCR inference: {e}")
        raise e
