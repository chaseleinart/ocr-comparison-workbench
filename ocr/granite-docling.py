from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# Global converter instance (lazy loaded)
_converter = None


def get_converter():
    """
    Get or create the DocumentConverter instance.
    
    Returns:
        DocumentConverter: The converter instance
    """
    global _converter
    if _converter is None:
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                ),
            }
        )
    return _converter


def run_ocr_inference(image_path: str) -> str:
    """
    Run OCR inference on an image using Granite Docling.
    
    Args:
        image_path: Path to the input image file
    
    Returns:
        str: The extracted text from the image
    """
    converter = get_converter()
    doc = converter.convert(source=image_path).document
    
    return doc.export_to_markdown()