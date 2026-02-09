"""
Memory Interface: Wrapper around MemoryStore for UI integration.
"""

import streamlit as st
from memory.store import MemoryStore


class MemoryInterface:
    """Interface for memory operations in the UI."""

    def __init__(self):
        self.store = MemoryStore()
        # Initialize persistent storage in session state
        if "memory_last_entry_id" not in st.session_state:
            st.session_state.memory_last_entry_id = None
        if "memory_last_entry" not in st.session_state:
            st.session_state.memory_last_entry = None

    @property
    def last_entry_id(self):
        """Get last entry ID from session state."""
        return st.session_state.get("memory_last_entry_id")

    @last_entry_id.setter
    def last_entry_id(self, value):
        """Set last entry ID in session state."""
        st.session_state.memory_last_entry_id = value
        print(f"🔹 [MemoryInterface] Set last_entry_id={value} in session_state")

    @property
    def last_entry(self):
        """Get last entry from session state."""
        return st.session_state.get("memory_last_entry")

    @last_entry.setter
    def last_entry(self, value):
        """Set last entry in session state."""
        st.session_state.memory_last_entry = value

    def store_interaction(self, query: str, topic: str, solution: str, verdict: str):
        """Store a problem-solution interaction and remember its id."""
        row_id = self.store.add(query, topic, solution, verdict)
        self.last_entry_id = row_id  # Uses property setter to store in session_state
        self.last_entry = {
            "query": query,
            "topic": topic,
            "solution": solution,
            "verdict": verdict,
        }
        print(f"✅ Stored interaction with ID {row_id}. Ready for feedback.")

    def feedback(self, ok: bool, comment: str = None):
        """
        Update feedback for the last stored interaction.
        No-op if no interaction was stored (e.g. cached reuse).
        """
        entry_id = self.last_entry_id
        print(f"🔍 [MemoryInterface.feedback] Checking entry_id: {entry_id}")
        if entry_id is None:
            print(
                "⚠️ WARNING: No entry to update feedback for (last_entry_id is None). Check session_state:"
            )
            print(
                f"   st.session_state.memory_last_entry_id = {st.session_state.get('memory_last_entry_id')}"
            )
            return
        print(
            f"✅ Storing feedback: ok={ok}, comment={comment[:50] if comment else 'None'}..., entry_id={entry_id}"
        )
        self.store.update_feedback(ok, comment, entry_id=entry_id)
        print(f"✅ Feedback stored successfully")

    def get_similar(self, query: str, top_k: int = 2, thresh: float = 0.75) -> list:
        """Return list of (similarity, past_query, past_solution)."""
        return self.store.similar(query, top_k=top_k, thresh=thresh)

    def get_stats(self) -> dict:
        """Return memory statistics."""
        return self.store.get_stats()
