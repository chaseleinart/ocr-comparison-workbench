import importlib.util
import os

# Get the directory of this __init__.py file
_ocr_dir = os.path.dirname(__file__)

# Import deepseek-ocr.py as deepseek_ocr module
_deepseek_spec = importlib.util.spec_from_file_location(
    "deepseek_ocr", os.path.join(_ocr_dir, "deepseek-ocr.py")
)
deepseek_ocr = importlib.util.module_from_spec(_deepseek_spec)
_deepseek_spec.loader.exec_module(deepseek_ocr)

# Import qwen-vl.py as qwen_vl module
_qwen_spec = importlib.util.spec_from_file_location(
    "qwen_vl",
    os.path.join(_ocr_dir, "qwen-vl.py")
)
qwen_vl = importlib.util.module_from_spec(_qwen_spec)
_qwen_spec.loader.exec_module(qwen_vl)

# Import granite-docling.py as granite_docling module
_granite_spec = importlib.util.spec_from_file_location(
    "granite_docling", os.path.join(_ocr_dir, "granite-docling.py")
)
granite_docling = importlib.util.module_from_spec(_granite_spec)
_granite_spec.loader.exec_module(granite_docling)

# Import chandra-ocr.py as chandra_ocr module
_chandra_spec = importlib.util.spec_from_file_location(
    "chandra_ocr", os.path.join(_ocr_dir, "chandra-ocr.py")
)
chandra_ocr = importlib.util.module_from_spec(_chandra_spec)
_chandra_spec.loader.exec_module(chandra_ocr)

# Import dots-ocr.py as dots_ocr module
_dots_spec = importlib.util.spec_from_file_location(
    "dots_ocr", os.path.join(_ocr_dir, "dots-ocr.py")
)
dots_ocr = importlib.util.module_from_spec(_dots_spec)
_dots_spec.loader.exec_module(dots_ocr)

# Export the functions directly from the loaded modules
load_model = deepseek_ocr.load_model
deepseek_ocr_inference = deepseek_ocr.run_ocr_inference
qwen_ocr_inference = qwen_vl.run_ocr_inference
granite_ocr_inference = granite_docling.run_ocr_inference
chandra_ocr_inference = chandra_ocr.run_ocr_inference
dots_ocr_inference = dots_ocr.run_ocr_inference
__all__ = [
    "load_model",
    "deepseek_ocr_inference",
    "qwen_ocr_inference",
    "granite_ocr_inference",
    "chandra_ocr_inference",
    "dots_ocr_inference",
]
