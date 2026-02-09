"""
OCR module using EasyOCR for image text extraction.
Confidence scoring and rule-based corrections from rules/ocr.json.
"""

import json
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image

_reader = None


def get_reader():
    """Get or create EasyOCR reader (singleton)."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def load_correction_rules():
    """Load OCR correction rules from rules/ocr.json."""
    rules_path = Path(__file__).parent.parent / "rules" / "ocr.json"
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load OCR rules: {e}")
        return {}


def apply_corrections(text: str, rules: dict) -> str:
    """Apply correction rules to OCR text."""
    corrected = text
    for wrong, correct in rules.items():
        corrected = corrected.replace(wrong, correct)
    return corrected


def run_ocr(image_file) -> tuple:
    """Extract text from image using EasyOCR. Returns (extracted_text, confidence)."""
    try:
        img = Image.open(image_file).convert("RGB")
        img_np = np.array(img)
        reader = get_reader()
        results = reader.readtext(img_np)
        if not results:
            return "No text detected in image", 0.5
        text_parts = [res[1] for res in results]
        confidences = [res[2] for res in results]
        raw_text = " ".join(text_parts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
        rules = load_correction_rules()
        corrected_text = apply_corrections(raw_text, rules)
        return corrected_text, avg_conf
    except Exception as e:
        return f"OCR failed: {str(e)}", 0.3
