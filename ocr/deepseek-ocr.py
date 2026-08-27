from unsloth import FastVisionModel
from transformers import AutoModel
import os

# Suppress Unsloth warning about uninitialized weights (position_ids is normal)
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"


def load_model(model_name="./deepseek_ocr", load_in_4bit=False):
    """
    Load the LoRA fine-tuned DeepSeek OCR model.

    Args:
        model_name: Path to the LoRA model directory (default: "lora_model")
        load_in_4bit: Whether to use 4bit quantization (default: False)

    Returns:
        tuple: (model, tokenizer)
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)  # Enable for inference!
    return model, tokenizer


def run_ocr_inference(
    model,
    tokenizer,
    image_file,
    output_path,
    prompt="<image>\nFree OCR. ",
    image_size=640,
    base_size=1024,
    crop_mode=True,
    save_results=True,
    test_compress=False,
):
    """
    Run OCR inference on an image using the loaded model.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        image_file: Path to the input image file
        output_path: Directory where results will be saved
        prompt: The prompt to use for OCR (default: "<image>\\nFree OCR. ")
        image_size: Image size for processing (default: 640)
        base_size: Base size for processing (default: 1024)
        crop_mode: Whether to use crop mode (default: True)
        save_results: Whether to save results to file (default: True)
        test_compress: Whether to test compression (default: False)

    Returns:
        The inference result
    """
    model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path=output_path,
        image_size=image_size,
        base_size=base_size,
        crop_mode=crop_mode,
        save_results=save_results,
        test_compress=test_compress,
    )
    result_file = os.path.join(output_path, "result.mmd")
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            ocr_result = f.read()
    else:
        ocr_result = "No text found in the image."
    return ocr_result