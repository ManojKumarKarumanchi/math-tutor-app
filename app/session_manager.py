"""
Session Manager: Handles user sessions, IDs, and state management.
"""

import uuid
from datetime import datetime

import streamlit as st


class SessionManager:
    """Manages user sessions and state for the Math Mentor application."""

    def __init__(self):
        self._initialize_session()

    def _initialize_session(self):
        """Initialize session state variables."""
        if "user_id" not in st.session_state:
            st.session_state.user_id = self._get_or_create_user_id()
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        if "hitl_approvals" not in st.session_state:
            st.session_state.hitl_approvals = {}
        if "agent_state" not in st.session_state:
            st.session_state.agent_state = {
                "current_problem": None,
                "parsed_data": None,
                "rag_context": None,
                "solution": None,
                "verification": None,
            }

    def _get_or_create_user_id(self) -> str:
        """Get or create persistent user ID."""
        if "persistent_user_id" not in st.session_state:
            user_id = f"user_{uuid.uuid4().hex[:8]}"
            st.session_state.persistent_user_id = user_id
        return st.session_state.persistent_user_id

    def get_user_id(self) -> str:
        """Get current user ID."""
        return st.session_state.user_id

    def get_session_id(self) -> str:
        """Get current session ID."""
        return st.session_state.session_id

    def new_session(self):
        """Start a new session."""
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.conversation_history = []
        st.session_state.hitl_approvals = {}
        st.session_state.agent_state = {
            "current_problem": None,
            "parsed_data": None,
            "rag_context": None,
            "solution": None,
            "verification": None,
        }

    def add_to_history(self, role: str, content: str, metadata: dict = None):
        """Add message to conversation history."""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        st.session_state.conversation_history.append(entry)

    def get_history(self, last_n: int = None) -> list:
        """Get conversation history, optionally last_n messages."""
        if last_n:
            return st.session_state.conversation_history[-last_n:]
        return st.session_state.conversation_history

    def set_agent_state(self, key: str, value):
        """Set agent state value."""
        st.session_state.agent_state[key] = value

    def get_agent_state(self, key: str):
        """Get agent state value."""
        return st.session_state.agent_state.get(key)

    def request_hitl_approval(self, stage: str, data: dict) -> bool:
        """
        Request human approval for a stage.
        stage: e.g. "ocr", "parser", "verifier".
        Returns True if already approved, False otherwise.
        """
        approval_key = f"hitl_approval_{stage}"
        if st.session_state.hitl_approvals.get(approval_key):
            return True
        return False

    def approve_hitl(self, stage: str):
        """Mark a HITL stage as approved."""
        approval_key = f"hitl_approval_{stage}"
        st.session_state.hitl_approvals[approval_key] = True

    def get_session_summary(self) -> dict:
        """Return session summary statistics."""
        return {
            "user_id": self.get_user_id(),
            "session_id": self.get_session_id(),
            "messages": len(st.session_state.conversation_history),
            "hitl_approvals": len(st.session_state.hitl_approvals),
        }
