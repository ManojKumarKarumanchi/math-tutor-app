"""
ASR module using Groq Whisper for audio transcription.
Math-specific phrase handling from rules/asr.json.
"""

import json
import os
from pathlib import Path

from groq import Groq


def load_correction_rules():
    """Load ASR correction rules from rules/asr.json."""
    rules_path = Path(__file__).parent.parent / "rules" / "asr.json"
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load ASR rules: {e}")
        return {}


def apply_corrections(text: str, rules: dict) -> str:
    """Apply correction rules to ASR transcript (math-specific phrases)."""
    corrected = text
    sorted_rules = sorted(rules.items(), key=lambda x: len(x[0]), reverse=True)
    for phrase, replacement in sorted_rules:
        corrected = corrected.replace(phrase, replacement)
    return corrected


def run_asr(audio_file) -> tuple:
    """Convert audio to text using Groq Whisper. Returns (transcript, confidence)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not found in environment variables", 0.0
    client = Groq(api_key=api_key)
    try:
        audio_bytes = audio_file.read()
        from config.models import AUDIO_MODEL

        transcript_resp = client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model=AUDIO_MODEL,
            response_format="verbose_json",
        )
        raw_text = transcript_resp.text
        rules = load_correction_rules()
        corrected_text = apply_corrections(raw_text, rules)
        return corrected_text, 0.85
    except Exception as e:
        return f"Transcription failed: {str(e)}", 0.3
