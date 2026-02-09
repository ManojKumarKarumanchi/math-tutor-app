"""
Main application entry point.
Initializes environment, knowledge base, and launches the Streamlit UI.
"""

import os

import streamlit as st
from dotenv import load_dotenv


def init_env():
    """Load environment variables and log API key status."""
    load_dotenv()
    from config.models import (
        GEMINI_MODELS,
        GROQ_MODELS,
        get_fallback_models,
    )

    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GEMINI_API_KEY")

    print("=" * 70)
    print("Math Mentor AI - Checking API Keys")
    print("=" * 70)
    print(f"\nGROQ_API_KEY:   {'SET' if groq_key else 'NOT SET'}")
    print(f"GEMINI_API_KEY: {'SET' if google_key else 'NOT SET'}")

    fallback_models = get_fallback_models()
    print(f"\nModel Fallback Chain: {len(fallback_models)} models available")
    # for i, model in enumerate(fallback_models, 1):
    #     provider = "Groq" if model.startswith("groq:") else "Gemini"
    #     print(f"   {i}. {model} ({provider})")

    if groq_key and not google_key:
        print("\nWARNING: Only Groq API key is set.")
        print("Add GEMINI_API_KEY for automatic fallback.")
    elif groq_key and google_key:
        print(f"\nBoth API keys set: {len(fallback_models)} models in fallback chain")
    elif not groq_key and not google_key:
        print(
            "\nERROR: No API keys set. Please add GROQ_API_KEY or GEMINI_API_KEY to .env"
        )
    print("=" * 70)
    print()


def init_knowledge_base():
    """Initialize RAG knowledge base if not already done."""
    try:
        from rag.load_kb import load_kb

        kb_path = "./rag/kb"
        if os.path.exists(kb_path) and os.listdir(kb_path):
            load_kb()
            print("Knowledge base loaded successfully")
        else:
            print("No knowledge base files found in ./rag/kb")
    except Exception as e:
        print(f"Could not load knowledge base: {e}")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Math Mentor AI",
        layout="wide",
        page_icon="🧮",
        initial_sidebar_state="expanded",
    )

    init_env()

    if "kb_loaded" not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            init_knowledge_base()
        st.session_state.kb_loaded = True

    from app.ui import app_main

    app_main()


if __name__ == "__main__":
    main()
