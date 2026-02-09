"""
Text normalization and cleaning utilities.

Standardizes text from all input sources (text, OCR, ASR).

NOTE: No semantic parsing or math reasoning here.
"""

import re


def process_text(raw: str) -> str:
    """
    Clean and normalize text input.

    Args:
        raw (str): Raw text from any source.

    Returns:
        str: Cleaned and normalized text.
    """
    if not raw:
        return ""

    # Trim and normalize whitespace
    text = raw.strip()
    text = re.sub(r"\s+", " ", text)

    # ---------- OCR SYMBOL NORMALIZATION ----------
    symbol_normalizations = [
        ("√", "sqrt"),
        ("∛", "cbrt"),
        ("×", "*"),
        ("✕", "*"),
        ("·", "*"),
        ("÷", "/"),
        ("—", "-"),
        ("–", "-"),
        ("−", "-"),
        ("≤", "<="),
        ("≥", ">="),
        ("≠", "!="),
        ("π", "pi"),
    ]
    for symbol, replacement in symbol_normalizations:
        text = text.replace(symbol, replacement)

    # ---------- ASR PHRASE NORMALIZATION ----------
    asr_replacements = {
        r"\bsquare root of\b": "sqrt",
        r"\bsquare root\b": "sqrt",
        r"\bcube root of\b": "cbrt",
        r"\bcube root\b": "cbrt",
        r"\braised to the power of\b": "^",
        r"\braised to\b": "^",
        r"\bto the power of\b": "^",
        r"\bmultiplied by\b": "*",
        r"\btimes\b": "*",
        r"\binto\b": "*",
        r"\bdivided by\b": "/",
        r"\bdivide by\b": "/",
        r"\bover\b": "/",
        r"\bplus\b": "+",
        r"\bminus\b": "-",
        r"\bequals\b": "=",
        r"\bis equal to\b": "=",
    }
    for pattern, repl in asr_replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # ---------- SPOKEN NUMBERS ----------
    spoken_numbers = {
        r"\bzero\b": "0",
        r"\bone\b": "1",
        r"\btwo\b": "2",
        r"\bthree\b": "3",
        r"\bfour\b": "4",
        r"\bfive\b": "5",
        r"\bsix\b": "6",
        r"\bseven\b": "7",
        r"\beight\b": "8",
        r"\bnine\b": "9",
        r"\bten\b": "10",
    }
    for pattern, repl in spoken_numbers.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # ---------- POWER NORMALIZATION ----------
    # Keep math-engine friendly
    text = text.replace("^", "**")

    # ---------- IMPLICIT MULTIPLICATION (SAFE CASES ONLY) ----------
    # 2x → 2*x
    text = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", text)
    # x2 → x*2
    text = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", text)
    # )x → )*x
    text = re.sub(r"\)([a-zA-Z0-9])", r")*\1", text)

    # ---------- BRACKET NORMALIZATION ----------
    text = text.replace("[", "(").replace("]", ")")
    text = text.replace("{", "(").replace("}", ")")

    return text
