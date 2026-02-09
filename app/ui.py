"""
Main UI module: Streamlit interface for Math Mentor AI.
"""

import json
import streamlit as st
from app.trace import TraceLogger
from app.memory_interface import MemoryInterface
from app.session_manager import SessionManager
from rag.ocr import run_ocr
from rag.asr import run_asr
from rag.text_processor import process_text
from agents.parser import parse_problem
from agents.router import route_topic
from agents.solver import solve_problem
from agents.verifier import verify_solution
from agents.explainer import explain_solution
from agents.guardrail import check_guardrails
from agents.web_search import search_web
from rag.retriever import retrieve_context, check_context_sufficiency


def _solution_to_markdown(solution) -> str:
    """Normalize solution to string for display (handles dict/error from solver)."""
    if isinstance(solution, dict):
        if "error" in solution:
            err = solution.get("error")
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return f"**Solver encountered an error:**\n\n{msg}"
        return json.dumps(solution, indent=2)
    return str(solution) if solution else ""


def _render_markdown_block(text: str, max_len: int = 2000) -> None:
    """Render a block of text as markdown (safe for long/raw content)."""
    if not text:
        return
    snippet = text[:max_len] + ("..." if len(text) > max_len else "")
    st.markdown(snippet)


def _render_kb_citations(rag_citations: list, rag_docs_fallback: list = None) -> None:
    """
    Render KB citations with source/path and Show preview (expander).
    rag_citations: list of dicts with keys content, source, name, category (from retriever).
    rag_docs_fallback: if no rag_citations, use this list of content strings (legacy).
    """
    if rag_citations:
        st.caption(
            f"Retrieved {len(rag_citations)} chunk(s) from local knowledge base (LanceDB)"
        )
        for i, cite in enumerate(rag_citations):
            source = cite.get("source") or cite.get("name") or f"Chunk {i+1}"
            label = f"**Source {i+1}** — `{source}`"
            with st.expander(f"📄 {label} — Show preview", expanded=False):
                _render_markdown_block(cite.get("content") or "", max_len=800)
            if i < len(rag_citations) - 1:
                st.divider()
    elif rag_docs_fallback:
        st.caption(
            f"Retrieved {len(rag_docs_fallback)} chunk(s) from local knowledge base (LanceDB)"
        )
        for i, doc in enumerate(rag_docs_fallback):
            st.markdown(f"**Source {i+1}**")
            _render_markdown_block(doc, max_len=600)
            if i < len(rag_docs_fallback) - 1:
                st.divider()
    else:
        st.caption("No documents found from the knowledge base (LanceDB).")


def _render_web_search_content(content: str) -> None:
    """Render web search results as neat markdown with expandable previews."""
    content = content.replace("Web Search Results:\n", "").strip()
    try:
        # Try to parse as structured data
        if "---" in content:
            # DuckDuckGo formatted results
            results = content.split("---")
            for i, result in enumerate(results, 1):
                result = result.strip()
                if not result:
                    continue

                # Extract title, URL, and content
                lines = result.split("\n")
                title = ""
                url = ""
                body_lines = []

                for line in lines:
                    if line.startswith("**[") and "]**" in line:
                        # Extract title
                        title_part = line.split("]**")[0].replace("**[", "").strip()
                        if "**" in title_part:
                            title = title_part.split("**")[0].strip()
                    elif line.startswith("**URL**:") or line.startswith("**URL**:"):
                        url = line.split(":", 1)[1].strip()
                    elif line.startswith("**Content**:") or line.startswith(
                        "**Content**:"
                    ):
                        body_lines.append(line.split(":", 1)[1].strip())
                    elif line and not line.startswith("**"):
                        body_lines.append(line)

                body = " ".join(body_lines) if body_lines else result

                # Show title/URL as clickable, with expandable preview
                if title or url:
                    display_title = title or url or f"Result {i}"
                    with st.expander(
                        f"🔗 {i}. {display_title} — Show preview", expanded=False
                    ):
                        if url:
                            st.markdown(f"**URL:** [{url}]({url})")
                        if body:
                            _render_markdown_block(body, max_len=1000)
                else:
                    with st.expander(f"🔗 Result {i} — Show preview", expanded=False):
                        _render_markdown_block(result, max_len=1000)

                if i < len(results):
                    st.divider()
        else:
            # Try JSON format
            data = json.loads(content)
            if isinstance(data, list):
                for i, item in enumerate(data, 1):
                    if isinstance(item, dict):
                        title = item.get("title", "Result")
                        href = item.get("href", "") or item.get("url", "")
                        body = item.get("body", "") or item.get("snippet", "")

                        with st.expander(
                            f"🔗 {i}. {title} — Show preview", expanded=False
                        ):
                            if href:
                                st.markdown(f"**URL:** [{href}]({href})")
                            if body:
                                _render_markdown_block(body, max_len=1000)
                    else:
                        with st.expander(
                            f"🔗 Result {i} — Show preview", expanded=False
                        ):
                            _render_markdown_block(str(item), max_len=1000)

                    if i < len(data):
                        st.divider()
            else:
                with st.expander("🔗 Web Search Result — Show preview", expanded=False):
                    _render_markdown_block(content, max_len=1000)
    except (json.JSONDecodeError, ValueError):
        # Plain text format - show in expander
        with st.expander("🔗 Web Search Result — Show preview", expanded=False):
            _render_markdown_block(content, max_len=1000)


def reset_to_home():
    """Reset all state and return to input page."""
    keys_to_clear = [
        "result_view_state",
        "verifier_pending_resume",
        "verifier_accepted_resume",
        "show_feedback",
        "feedback_success_msg",
        "clarification_submitted",
        "clarified_text",
        "clarified_parsed",
        "ocr_extracted",
        "ocr_confidence",
        "asr_transcript",
        "asr_confidence",
        "hitl_stage",
        "audio_bytes",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.session_state.hitl_stage = "input"
    st.rerun()


def app_main():
    """Main Streamlit UI"""
    st.title("🧮 Math Mentor AI")
    st.caption("Multimodal math solver with RAG + Agents + HITL + Memory")

    if "trace" not in st.session_state:
        st.session_state.trace = TraceLogger()
    if "memory" not in st.session_state:
        st.session_state.memory = MemoryInterface()
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    trace = st.session_state.trace
    memory = st.session_state.memory
    session_mgr = st.session_state.session_manager

    # ========== SIDEBAR ==========
    with st.sidebar:
        # Home button at top of sidebar
        if st.button(
            "🏠 Home", type="primary", use_container_width=True, key="sidebar_home"
        ):
            reset_to_home()
        st.divider()
        st.subheader("🤖 Model Selection")
        from config.models import (
            get_available_models,
            get_model_display_name,
        )

        available = get_available_models()
        if available:
            provider = st.selectbox(
                "Provider",
                options=list(available.keys()),
                key="provider_select",
            )
            if provider and provider in available:
                model_options = available[provider]
                display_options = [get_model_display_name(m) for m in model_options]
                selected_idx = st.selectbox(
                    "Model",
                    options=range(len(model_options)),
                    format_func=lambda i: display_options[i],
                    key="model_select",
                )
                st.session_state.selected_model = model_options[selected_idx]
                st.caption(f"✅ Selected: `{st.session_state.selected_model}`")
        else:
            st.error("⚠️ No API keys configured")
            st.info(
                "**Setup Instructions:**\n"
                "1. Copy `.env.example` to `.env`\n"
                "2. Add at least one API key:\n"
                "   - `GROQ_API_KEY` (Llama models + Whisper)\n"
                "   - `GEMINI_API_KEY` (Google Gemini models)\n"
                "3. Restart the app"
            )
            st.session_state.selected_model = None

        st.divider()
        st.subheader("🔍 Agent Trace")
        logs = trace.get_logs()
        if logs:
            for step in logs:
                st.markdown(f"**{step['agent']}**: {step['message']}")
        else:
            st.info("No activity yet")
        st.divider()
        st.subheader("👤 Session Info")
        session_info = session_mgr.get_session_summary()
        st.caption(f"User ID: `{session_info['user_id']}`")
        st.caption(f"Session: `{session_info['session_id'][:8]}...`")
        if st.button(
            "🔄 New Session", use_container_width=True, key="sidebar_new_session"
        ):
            session_mgr.new_session()
            st.rerun()
        st.divider()
        st.subheader("📊 Memory Stats")
        stats = memory.get_stats()
        st.metric("Total Problems", stats["total"])
        col1, col2 = st.columns(2)
        col1.metric("👍 Positive", stats["positive_feedback"])
        col2.metric("👎 Negative", stats["negative_feedback"])

    # ========== PENDING CLARIFICATION: run orchestration immediately after Submit Clarification ==========
    # This runs first so we don't depend on the form or submit button block (which can fail to trigger).
    if st.session_state.get("clarification_submitted") and st.session_state.get(
        "clarified_text"
    ):
        clarified_text = st.session_state.clarified_text
        # Clear immediately so we don't re-enter on next run
        st.session_state.clarification_submitted = False
        if "clarified_text" in st.session_state:
            del st.session_state.clarified_text
        if "clarified_parsed" in st.session_state:
            del st.session_state.clarified_parsed
        processed = process_text(clarified_text)
        selected_model = st.session_state.get("selected_model")
        orchestrate(processed, trace, memory, session_mgr, selected_model, None, None)
        # Sidebar already drawn at top of app_main(); stop so result stays visible
        st.stop()

    # ========== CHECK FEEDBACK STATE FIRST (before verifier sections) ==========
    # If user clicked "Incorrect" and wants to provide feedback, show comment box
    if st.session_state.get("show_feedback") and st.session_state.get(
        "result_view_state"
    ):
        rv = st.session_state.get("result_view_state")
        memory = st.session_state.get("memory")

        # Show solution and explanation first
        st.markdown("---")
        st.success("✅ Solution Complete!")
        if rv.get("reused_from_memory"):
            st.caption(
                "📌 This question was previously solved; showing cached solution (not counted again)."
            )
        st.markdown("---")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📝 Solution")
        with col2:
            st.metric("Confidence", f"{rv.get('confidence', 0)*100:.1f}%")

        solution_display_rv = rv.get("solution_display") or ""
        # Solution in blue-tinted container with proper markdown
        st.markdown(
            f'<div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; border-left: 4px solid #2196f3; margin: 10px 0;">'
            f'<div style="color: #1565c0; font-weight: bold; margin-bottom: 8px;">Solution:</div>',
            unsafe_allow_html=True,
        )
        st.markdown(solution_display_rv)
        st.markdown("</div>", unsafe_allow_html=True)

        explanation_rv = rv.get("explanation") or ""
        if explanation_rv.strip() != solution_display_rv.strip():
            st.subheader("💡 Explanation")
            # Explanation in green-tinted container with proper markdown
            st.markdown(
                f'<div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; border-left: 4px solid #4caf50; margin: 10px 0;">'
                f'<div style="color: #2e7d32; font-weight: bold; margin-bottom: 8px;">Explanation:</div>',
                unsafe_allow_html=True,
            )
            st.markdown(explanation_rv)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("✅ Feedback")
        st.caption("Your feedback helps improve the system.")
        comment = st.text_area(
            "What was wrong? (optional)", key="feedback_comment_early", height=100
        )
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button(
                "Submit Feedback", key="feedback_submit_early", type="primary"
            ):
                if memory:
                    memory.feedback(False, comment)
                st.session_state["feedback_success_msg"] = "Feedback recorded!"
                st.session_state.pop("result_view_state", None)
                st.session_state.pop("show_feedback", None)
                reset_to_home()
        with c2:
            if st.button("Cancel", key="feedback_cancel_early"):
                st.session_state["show_feedback"] = False
                st.rerun()
        with c3:
            if st.button(
                "🏠 Home", key="feedback_home_comment_early", use_container_width=True
            ):
                reset_to_home()
        st.stop()

    # ========== VERIFIER ACCEPTED: show solution after user clicked "Accept & show solution" (check first so solution shows) ==========
    if st.session_state.get("verifier_accepted_resume"):
        resume = st.session_state.verifier_accepted_resume
        del st.session_state["verifier_accepted_resume"]

        # Add Home button at top of result view
        if st.button(
            "🏠 Home", key="home_verifier_accepted", use_container_width=False
        ):
            reset_to_home()
        st.markdown("---")
        parsed = resume["parsed"]
        topic = resume["topic"]
        docs = resume["docs"]
        solution = resume["solution"]
        solution_display = resume["solution_display"]
        tool_calls = resume["tool_calls"]
        verdict = resume["verdict"]
        confidence = resume["confidence"]
        memory_context = resume.get("memory_context") or []
        session_mgr.set_agent_state("tool_calls", tool_calls)
        session_mgr.set_agent_state("parsed", parsed)
        session_mgr.set_agent_state("topic", topic)
        trace.log("Explainer", "Generating explanation...")
        selected_model = st.session_state.get("selected_model")
        with st.spinner("Generating explanation..."):
            explanation = explain_solution(
                solution_display, docs, parsed, selected_model
            )
        trace.log("Explainer", "✅ Explanation generated")
        memory.store_interaction(
            query=parsed.get("problem_text"),
            topic=topic,
            solution=solution_display,
            verdict=verdict,
        )
        st.markdown("---")
        st.success("✅ Solution Complete!")
        with st.expander("🔍 Pipeline execution trace", expanded=False):
            for step in trace.get_logs():
                st.markdown(f"- **{step['agent']}**: {step['message']}")
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📝 Solution")
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")

        # Solution in blue-tinted container
        st.markdown(
            f'<div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; border-left: 4px solid #2196f3; margin: 10px 0;">'
            f'<div style="color: #1565c0; font-weight: bold; margin-bottom: 8px;">Solution:</div>',
            unsafe_allow_html=True,
        )
        st.markdown(solution_display)
        st.markdown("</div>", unsafe_allow_html=True)

        if explanation and explanation.strip() != (solution_display or "").strip():
            st.subheader("💡 Explanation")
            # Explanation in green-tinted container
            st.markdown(
                f'<div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; border-left: 4px solid #4caf50; margin: 10px 0;">'
                f'<div style="color: #2e7d32; font-weight: bold; margin-bottom: 8px;">Explanation:</div>',
                unsafe_allow_html=True,
            )
            st.markdown(explanation)
            st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("🧠 Agent thoughts & reasoning", expanded=False):
            st.markdown("#### Parser Agent")
            st.json(parsed)
            st.markdown("---")
            st.markdown("#### Router Agent")
            st.markdown(f"**Topic:** `{topic}`")
            st.markdown("---")
            st.markdown("#### Solver Agent")
            if tool_calls:
                st.markdown(f"**Tool calls:** {len(tool_calls)} SymPy operation(s)")
                for i, tc in enumerate(tool_calls, 1):
                    st.code(
                        f"{i}. {tc.get('tool', 'unknown')}({tc.get('arguments', {})})",
                        language="text",
                    )
            else:
                st.markdown("No tool calls (direct reasoning).")
            st.markdown("---")
            st.markdown("#### Verifier Agent")
            st.markdown(
                f"**Verdict:** {verdict}  \n**Confidence:** {confidence*100:.1f}%"
            )
            if memory_context:
                st.markdown("---")
                st.markdown("#### Memory")
                st.markdown(
                    f"Used {len(memory_context)} similar past problem(s) as context."
                )
        rag_docs = [d for d in docs if not d.startswith("Web Search Results:")]
        web_docs = [d for d in docs if d.startswith("Web Search Results:")]
        rag_citations_resume = resume.get("rag_citations") or []

        # Always show KB citations if available
        if rag_docs or rag_citations_resume:
            with st.expander("📚 Knowledge base citations", expanded=False):
                _render_kb_citations(rag_citations_resume, rag_docs_fallback=rag_docs)

        # Always show web search citations (run in parallel, not conditional)
        with st.expander("🌐 Web search citations", expanded=False):
            if web_docs:
                st.caption(
                    "Retrieved from DuckDuckGo (searched in parallel with knowledge base)"
                )
                for doc in web_docs:
                    _render_web_search_content(doc)
            else:
                st.caption("No web search results available")
        # ========== SIMILAR PAST PROBLEMS (Always show at bottom) ==========
        st.markdown("---")
        if memory_context:
            st.subheader("🧠 Similar Past Problems")
            st.caption(
                "Retrieved from memory by embedding similarity (learning from past solutions)"
            )
            for i, (sim, past_q, past_sol) in enumerate(memory_context, 1):
                with st.expander(
                    f"📌 Problem {i} — Similarity: {sim*100:.1f}%", expanded=False
                ):
                    st.markdown(f"**Question:** {past_q}")
                    st.markdown("**Solution:**")
                    _render_markdown_block(past_sol, max_len=800)
                if i < len(memory_context):
                    st.divider()
        else:
            st.subheader("🧠 Similar Past Problems")
            st.caption(
                "No similar problems found in memory. This problem will be stored for future reference."
            )
        st.markdown("---")
        if st.session_state.get("feedback_success_msg"):
            st.success(st.session_state["feedback_success_msg"])
            st.session_state.pop("feedback_success_msg", None)
        st.subheader("✅ Feedback")
        st.caption("Your feedback helps improve the system.")
        show_fb = st.session_state.get("show_feedback", False)
        if not show_fb:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button(
                    "👍 Correct",
                    key="feedback_correct_resume",
                    use_container_width=True,
                ):
                    memory.feedback(True)
                    st.session_state["feedback_success_msg"] = "Thanks for confirming!"
                    reset_to_home()
            with col2:
                if st.button(
                    "👎 Incorrect",
                    key="feedback_incorrect_resume",
                    use_container_width=True,
                ):
                    st.session_state["show_feedback"] = True
                    st.rerun()
            with col3:
                if st.button(
                    "🏠 Home", key="feedback_home_resume", use_container_width=True
                ):
                    reset_to_home()
        else:
            comment = st.text_area(
                "What was wrong? (optional)", key="feedback_comment_resume"
            )
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("Submit Feedback", key="feedback_submit_resume"):
                    memory.feedback(False, comment)
                    st.session_state["feedback_success_msg"] = "Feedback recorded!"
                    reset_to_home()
            with c2:
                if st.button("Cancel", key="feedback_cancel_resume"):
                    st.session_state["show_feedback"] = False
                    st.rerun()
            with c3:
                if st.button(
                    "🏠 Home",
                    key="feedback_home_comment_resume",
                    use_container_width=True,
                ):
                    reset_to_home()
        # Clear pending if present (cleanup)
        if "verifier_pending_resume" in st.session_state:
            del st.session_state["verifier_pending_resume"]
        # Sidebar already drawn at top of app_main()
        st.stop()

    # ========== VERIFIER PENDING: show HITL again so "Accept" click is processed after rerun ==========
    if st.session_state.get("verifier_pending_resume"):
        pending = st.session_state.verifier_pending_resume
        verdict = pending.get("verdict", "incorrect")
        confidence = pending.get("confidence", 0)
        st.markdown("---")
        if verdict == "incorrect":
            st.error("❌ **Verifier Assessment: Solution appears INCORRECT**")
            st.warning(f"Confidence: {confidence*100:.1f}%")
            st.markdown("The verifier identified potential errors. You can:")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button(
                    "❌ Reject & Stop",
                    key="verifier_reject_pending",
                    use_container_width=True,
                ):
                    reset_to_home()
            with col2:
                if st.button(
                    "⚠️ Accept & show solution",
                    key="verifier_accept_pending_incorrect",
                    use_container_width=True,
                    type="secondary",
                ):
                    st.session_state.verifier_accepted_resume = dict(pending)
                    del st.session_state["verifier_pending_resume"]
                    st.rerun()
            with col3:
                if st.button(
                    "🏠 Home", key="verifier_home_pending", use_container_width=True
                ):
                    reset_to_home()
        else:
            st.warning(
                f"⚠️ **Verifier is uncertain** (Verdict: {verdict}, Confidence: {confidence*100:.1f}%)"
            )
            st.caption("The verifier cannot confirm correctness with high confidence.")
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button(
                    "✅ Accept & show solution",
                    type="primary",
                    key="verifier_accept_pending",
                ):
                    st.session_state.verifier_accepted_resume = dict(pending)
                    del st.session_state["verifier_pending_resume"]
                    st.rerun()
            with col2:
                if st.button(
                    "🏠 Home", key="verifier_home_uncertain", use_container_width=True
                ):
                    reset_to_home()
            st.info("💡 Tip: Try rephrasing your question or providing more context.")
        st.stop()

    # ========== RESULT VIEW: re-show solution + feedback after feedback-button rerun ==========
    # (Avoids redirect to form when user clicks Correct/Incorrect; show same page + comment UI or success.)
    rv = st.session_state.get("result_view_state")
    if rv:
        memory = st.session_state.get("memory")

        # Add Home button at top of result view
        if st.button("🏠 Home", key="home_result_view", use_container_width=False):
            reset_to_home()
        st.markdown("---")

        st.success("✅ Solution Complete!")
        if rv.get("reused_from_memory"):
            st.caption(
                "📌 This question was previously solved; showing cached solution (not counted again)."
            )
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📝 Solution")
        with col2:
            st.metric("Confidence", f"{rv.get('confidence', 0)*100:.1f}%")

        solution_display_rv = rv.get("solution_display") or ""
        # Solution in blue-tinted container
        st.markdown(
            f'<div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; border-left: 4px solid #2196f3; margin: 10px 0;">'
            f'<div style="color: #1565c0; font-weight: bold; margin-bottom: 8px;">Solution:</div>',
            unsafe_allow_html=True,
        )
        st.markdown(solution_display_rv)
        st.markdown("</div>", unsafe_allow_html=True)

        explanation = rv.get("explanation") or ""
        if explanation.strip() != solution_display_rv.strip():
            st.subheader("💡 Explanation")
            # Explanation in green-tinted container
            st.markdown(
                f'<div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; border-left: 4px solid #4caf50; margin: 10px 0;">'
                f'<div style="color: #2e7d32; font-weight: bold; margin-bottom: 8px;">Explanation:</div>',
                unsafe_allow_html=True,
            )
            st.markdown(explanation)
            st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("🧠 Agent thoughts & reasoning", expanded=False):
            st.markdown("#### Parser Agent")
            st.json(rv.get("parsed") or {})
            st.markdown("---")
            st.markdown("#### Router Agent")
            st.markdown(f"**Topic:** `{rv.get('topic', '')}`")
            st.markdown("---")
            st.markdown("#### Solver Agent")
            tool_calls_data = rv.get("tool_calls") or []
            if tool_calls_data:
                st.markdown(
                    f"**Tool calls:** {len(tool_calls_data)} SymPy operation(s)"
                )
                for i, tc in enumerate(tool_calls_data, 1):
                    args = tc.get("arguments", {})
                    st.code(
                        f"{i}. {tc.get('tool', 'unknown')}({args})", language="text"
                    )
            else:
                st.markdown("No tool calls (direct reasoning).")
            st.markdown("---")
            st.markdown("#### Verifier Agent")
            st.markdown(
                f"**Verdict:** {rv.get('verdict', '')}  \n**Confidence:** {rv.get('confidence', 0)*100:.1f}%"
            )
            if rv.get("memory_context"):
                st.markdown("---")
                st.markdown("#### Memory")
                st.markdown(
                    f"Used {len(rv['memory_context'])} similar past problem(s) as context."
                )
        docs = rv.get("docs") or []
        rag_docs = [d for d in docs if not str(d).startswith("Web Search Results:")]
        web_docs = [d for d in docs if str(d).startswith("Web Search Results:")]

        # Always show KB citations if available
        if rag_docs or rv.get("rag_citations"):
            with st.expander("📚 Knowledge base citations", expanded=False):
                _render_kb_citations(
                    rv.get("rag_citations") or [], rag_docs_fallback=rag_docs
                )

        # Always show web search citations (run in parallel, not conditional)
        with st.expander("🌐 Web search citations", expanded=False):
            if web_docs:
                st.caption(
                    "Retrieved from DuckDuckGo (searched in parallel with knowledge base)"
                )
                for doc in web_docs:
                    _render_web_search_content(doc)
            else:
                st.caption("No web search results available")
        # ========== SIMILAR PAST PROBLEMS (Always show at bottom) ==========
        st.markdown("---")
        memory_context_rv = rv.get("memory_context") or []
        if memory_context_rv:
            st.subheader("🧠 Similar Past Problems")
            st.caption(
                "Retrieved from memory by embedding similarity (learning from past solutions)"
            )
            for i, (sim, past_q, past_sol) in enumerate(memory_context_rv, 1):
                with st.expander(
                    f"📌 Problem {i} — Similarity: {sim*100:.1f}%", expanded=False
                ):
                    st.markdown(f"**Question:** {past_q}")
                    st.markdown("**Solution:**")
                    _render_markdown_block(past_sol, max_len=800)
                if i < len(memory_context_rv):
                    st.divider()
        else:
            st.subheader("🧠 Similar Past Problems")
            st.caption(
                "No similar problems found in memory. This problem will be stored for future reference."
            )
        st.markdown("---")
        if st.session_state.get("feedback_success_msg"):
            st.success(st.session_state["feedback_success_msg"])
            st.session_state.pop("feedback_success_msg", None)
        st.subheader("✅ Feedback")
        st.caption("Your feedback helps improve the system.")
        show_fb = st.session_state.get("show_feedback", False)
        if not show_fb:
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "👍 Correct", key="feedback_correct_rv", use_container_width=True
                ):
                    if memory:
                        memory.feedback(True)
                    st.session_state["feedback_success_msg"] = "Thanks for confirming!"
                    st.session_state.pop("result_view_state", None)
                    st.session_state.pop("show_feedback", None)
                    st.rerun()
            with col2:
                if st.button(
                    "👎 Incorrect",
                    key="feedback_incorrect_rv",
                    use_container_width=True,
                ):
                    st.session_state["show_feedback"] = True
                    st.rerun()
        else:
            st.markdown("---")
            st.subheader("💬 Your Feedback")
            comment = st.text_area(
                "What was wrong? (optional)", key="feedback_comment_rv", height=120
            )
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button(
                    "✅ Submit Feedback", key="feedback_submit_rv", type="primary"
                ):
                    if memory:
                        memory.feedback(False, comment)
                    st.session_state["feedback_success_msg"] = (
                        "✅ Feedback recorded! Thank you for helping us improve."
                    )
                    st.session_state.pop("result_view_state", None)
                    st.session_state.pop("show_feedback", None)
                    st.rerun()
            with c2:
                if st.button("⏮️ Back", key="feedback_cancel_rv"):
                    st.session_state["show_feedback"] = False
                    st.rerun()
            with c3:
                if st.button("🏠 Home", key="feedback_home_rv"):
                    reset_to_home()
        st.stop()

    # Show one-time success message from feedback (then clear so form shows)
    if st.session_state.get("feedback_success_msg"):
        st.success(st.session_state["feedback_success_msg"])
        st.session_state.pop("feedback_success_msg", None)

    # ========== Multi-modal input: text, image, and audio in one form ==========
    st.subheader("Input your math problem")

    # Initialize HITL workflow states
    if "hitl_stage" not in st.session_state:
        st.session_state.hitl_stage = (
            "input"  # Stages: input -> ocr_review -> asr_review -> ready
        )
    if "ocr_extracted" not in st.session_state:
        st.session_state.ocr_extracted = None
    if "ocr_confidence" not in st.session_state:
        st.session_state.ocr_confidence = None
    if "asr_transcript" not in st.session_state:
        st.session_state.asr_transcript = None
    if "asr_confidence" not in st.session_state:
        st.session_state.asr_confidence = None

    user_text = st.text_area(
        "Enter text (optional)",
        height=120,
        placeholder="e.g., Solve x^2 - 5x + 6 = 0",
        help="Type the problem, and/or add an image or voice recording below.",
        key="user_text_input",
    )

    col_img, col_audio = st.columns(2)
    with col_img:
        image_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg"],
            help="Photo of a handwritten or printed problem. (Max size: 5MB)",
            key="image_uploader",
        )
        if image_file is not None and image_file.size > 5 * 1024 * 1024:
            st.warning("⚠️ File size exceeds 5MB. Please upload a smaller image.")
            image_file = None
    with col_audio:
        audio_value = st.audio_input(
            "Record a voice message",
            help="Record your question with the microphone. Then click 'Transcribe' to get text (Groq Whisper).",
            key="audio_recorder",
        )

    # Persist audio bytes so we can transcribe after rerun (st.audio_input often clears)
    if audio_value is not None:
        st.session_state.audio_bytes = audio_value.read()
        audio_value.seek(0)

    # ========== HITL STAGE 1: OCR EXTRACTION & REVIEW ==========
    if image_file is not None and st.session_state.hitl_stage == "input":
        if st.button(
            "📸 Extract Text from Image", type="secondary", key="ocr_extract_btn"
        ):
            with st.spinner("🔍 Extracting text from image..."):
                extracted, conf = run_ocr(image_file)
                st.session_state.ocr_extracted = extracted or ""
                st.session_state.ocr_confidence = conf
                st.session_state.hitl_stage = "ocr_review"
                st.rerun()

    # Show OCR review interface if in ocr_review stage
    if st.session_state.hitl_stage == "ocr_review":
        st.markdown("---")
        st.subheader("📸 OCR Extraction Review (HITL)")

        conf_pct = st.session_state.ocr_confidence * 100
        conf_color = "🟢" if conf_pct >= 70 else "🟡" if conf_pct >= 50 else "🔴"
        st.markdown(f"{conf_color} **Confidence: {conf_pct:.1f}%**")

        if conf_pct < 70:
            st.warning(
                "⚠️ Low OCR confidence detected. Please review and edit the extracted text carefully."
            )

        # Editable OCR text
        ocr_edited = st.text_area(
            "Extracted text (review and edit if needed):",
            value=st.session_state.ocr_extracted,
            height=150,
            key="ocr_edit_area",
            help="Edit the extracted text to correct any OCR errors before proceeding.",
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Approve OCR Text", type="primary", key="ocr_approve_btn"):
                st.session_state.ocr_extracted = ocr_edited
                st.session_state.hitl_stage = "input"  # Move back to input stage
                st.success("✅ OCR text approved!")
                st.rerun()

        with col2:
            if st.button(
                "❌ Reject & Re-extract", type="secondary", key="ocr_reject_btn"
            ):
                st.session_state.ocr_extracted = None
                st.session_state.ocr_confidence = None
                st.session_state.hitl_stage = "input"
                st.warning("OCR rejected. Please upload a clearer image.")
                st.rerun()

    # ========== HITL STAGE 2: ASR TRANSCRIPTION & REVIEW ==========
    # Show Transcribe when we have audio (current widget or persisted bytes)
    has_audio = audio_value is not None or st.session_state.get("audio_bytes")
    if has_audio and st.session_state.hitl_stage == "input":
        if st.button(
            "🎤 Transcribe Audio (Groq Whisper)",
            type="secondary",
            key="asr_transcribe_btn",
        ):
            with st.spinner("🎧 Transcribing with Groq Whisper..."):
                import io

                source = (
                    audio_value
                    if audio_value is not None
                    else io.BytesIO(st.session_state.audio_bytes)
                )
                transcript, conf = run_asr(source)
                st.session_state.asr_transcript = transcript or ""
                st.session_state.asr_confidence = conf
                st.session_state.hitl_stage = "asr_review"
                st.rerun()

    # Show ASR review interface if in asr_review stage
    if st.session_state.hitl_stage == "asr_review":
        st.markdown("---")
        st.subheader("🎤 Audio Transcription Review (HITL)")

        conf_pct = st.session_state.asr_confidence * 100
        conf_color = "🟢" if conf_pct >= 70 else "🟡" if conf_pct >= 50 else "🔴"
        st.markdown(f"{conf_color} **Confidence: {conf_pct:.1f}%**")

        if conf_pct < 70:
            st.warning(
                "⚠️ Low transcription confidence detected. Please review and edit the transcript carefully."
            )

        # Editable ASR text
        asr_edited = st.text_area(
            "Transcription (review and edit if needed):",
            value=st.session_state.asr_transcript,
            height=150,
            key="asr_edit_area",
            help="Edit the transcription to correct any speech recognition errors before proceeding.",
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "✅ Approve Transcription", type="primary", key="asr_approve_btn"
            ):
                st.session_state.asr_transcript = asr_edited
                st.session_state.hitl_stage = "input"  # Move back to input stage
                st.success("✅ Transcription approved!")
                st.rerun()

        with col2:
            if st.button(
                "❌ Reject & Re-record", type="secondary", key="asr_reject_btn"
            ):
                st.session_state.asr_transcript = None
                st.session_state.asr_confidence = None
                st.session_state.hitl_stage = "input"
                st.warning("Transcription rejected. Please record again more clearly.")
                st.rerun()

    # ========== FINAL SUBMIT BUTTON ==========
    st.markdown("---")

    # Show what will be submitted
    has_text = (user_text or "").strip()
    has_ocr = bool(st.session_state.ocr_extracted)
    has_asr = bool(st.session_state.asr_transcript)
    preview_parts = []
    if has_text:
        preview_parts.append(f"**Text:** {user_text}")
    if has_ocr:
        preview_parts.append(f"**OCR (approved):** {st.session_state.ocr_extracted}")
    if has_asr:
        preview_parts.append(f"**Audio (approved):** {st.session_state.asr_transcript}")

    if preview_parts:
        with st.expander("📋 Preview combined input", expanded=True):
            for part in preview_parts:
                st.markdown(part)

    # Text-only: no HITL needed — user can submit directly
    if (
        st.session_state.hitl_stage == "input"
        and has_text
        and not has_ocr
        and not has_asr
    ):
        st.caption(
            "✅ You can submit your text directly below. No approval step needed for typed/pasted text."
        )

    # Submit when in input stage (text only, or after OCR/ASR approval)
    # OR auto-submit if we have a clarified problem from parser HITL
    should_auto_submit = st.session_state.get("clarification_submitted", False)

    if st.session_state.hitl_stage == "input":
        if (
            st.button("🚀 Submit Problem", type="primary", key="final_submit_btn")
            or should_auto_submit
        ):
            # Clear result view so we don't re-show previous solution after new submit
            st.session_state.pop("result_view_state", None)
            st.session_state.pop("show_feedback", None)
            # Check if this is an auto-submit from clarification
            if should_auto_submit:
                # Use clarified text
                combined = st.session_state.clarified_text
                ocr_conf = None
                asr_conf = None
                # Clear the clarification flags
                st.session_state.clarification_submitted = False
                if "clarified_text" in st.session_state:
                    del st.session_state.clarified_text
                if "clarified_parsed" in st.session_state:
                    del st.session_state.clarified_parsed
            else:
                # Normal submission
                parts = []
                ocr_conf = None
                asr_conf = None

                if (user_text or "").strip():
                    parts.append(user_text.strip())

                if st.session_state.ocr_extracted:
                    parts.append(st.session_state.ocr_extracted.strip())
                    ocr_conf = st.session_state.ocr_confidence

                if st.session_state.asr_transcript:
                    parts.append(st.session_state.asr_transcript.strip())
                    asr_conf = st.session_state.asr_confidence

                combined = "\n\n".join(parts).strip() if parts else ""

            if not combined:
                st.warning(
                    "⚠️ Please provide some input: type text, upload an image, or record voice."
                )
            else:
                selected_model = st.session_state.get("selected_model")
                if not selected_model:
                    st.error(
                        "⚠️ No model selected. Please configure API keys in `.env` and select a model from the sidebar."
                    )
                else:
                    processed = process_text(combined)
                    orchestrate(
                        processed,
                        trace,
                        memory,
                        session_mgr,
                        selected_model,
                        ocr_conf,
                        asr_conf,
                    )

                # Clear all HITL session state after successful submission
                for key in [
                    "ocr_extracted",
                    "ocr_confidence",
                    "asr_transcript",
                    "asr_confidence",
                    "hitl_stage",
                    "audio_bytes",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.hitl_stage = "input"


def orchestrate(
    processed: str,
    trace: TraceLogger,
    memory: MemoryInterface,
    session_mgr: SessionManager,
    selected_model: str = None,
    ocr_conf: float = None,
    asr_conf: float = None,
):
    """
    Main orchestration pipeline for solving problems.
    selected_model: User-selected model ID from sidebar (optional).
    """

    try:
        # Clear previous results
        trace.clear()

        print(f"\n🚀 Starting orchestration for: {processed[:50]}...")

        # ========== STEP 0: GUARDRAIL ==========
        trace.log("Guardrail", "Checking input safety and scope...")
        with st.spinner("Checking guardrails..."):
            approved, reason = check_guardrails(processed, selected_model)

        if not approved:
            st.error(f"❌ Request Rejected: {reason}")
            st.stop()

        trace.log("Guardrail", "✅ Approved")

        # ========== STEP 1: PARSE ==========
        trace.log("Parser", "Parsing problem structure...")
        with st.spinner("Parsing problem..."):
            parsed = parse_problem(processed, selected_model)

        print(f"✅ Parsed: {parsed}")

        # Check for rate limit error in parsed response
        if parsed.get("error"):
            error_info = parsed["error"]
            if isinstance(error_info, dict) and error_info.get("type") == "rate_limit":
                st.error("❌ **Rate Limit Exceeded**")
                st.warning(
                    error_info.get("message", "Rate limit exceeded on all models.")
                )
                st.info(
                    "💡 **Suggestions:**\n"
                    "- Wait a few minutes and try again\n"
                    "- Add a Gemini API key (GEMINI_API_KEY) for automatic fallback\n"
                    "- Check https://console.groq.com/settings/billing for usage\n"
                    "- Try a different Groq model (see https://console.groq.com/docs/models)"
                )
                return
            elif error_info:
                st.error(
                    f"❌ Parser error: {error_info if isinstance(error_info, str) else error_info.get('message', 'Unknown error')}"
                )
                return

        # ========== HITL: Parser Clarification ==========
        if parsed.get("needs_clarification"):
            st.markdown("---")
            st.warning("⚠️ **Parser detected ambiguity**")
            st.caption(
                f"**Issue**: {parsed.get('clarification_reason', 'Problem statement is unclear')}"
            )

            # Show what was parsed
            with st.expander("📋 What the parser understood", expanded=True):
                st.markdown(f"**Topic**: {parsed.get('topic', 'unknown')}")
                st.markdown(
                    f"**Original text**: {parsed.get('problem_text', processed)}"
                )

            # Initialize clarification state
            if "clarification_submitted" not in st.session_state:
                st.session_state.clarification_submitted = False

            clarification = st.text_area(
                "Please clarify or rephrase your question:",
                placeholder="e.g., Find where the function r(x) = (x+1)(x-2) is increasing or decreasing",
                height=100,
                key="clarification_input_text",
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(
                    "✅ Submit Clarification", type="primary", key="clarify_submit"
                ):
                    if clarification.strip():
                        # Re-parse with clarification
                        with st.spinner("Re-parsing with clarification..."):
                            new_processed = clarification.strip()
                            selected_model = st.session_state.get("selected_model")
                            new_parsed = parse_problem(new_processed, selected_model)

                            if new_parsed.get("needs_clarification"):
                                st.error(
                                    "❌ Still unclear. Please try rephrasing more clearly."
                                )
                                st.caption(
                                    "💡 Tip: Use standard math notation like r(x) = ... and state clearly what you want to find."
                                )
                                st.stop()
                            else:
                                # Success! Update the processed text and parsed result
                                st.session_state.clarification_submitted = True
                                st.session_state.clarified_text = new_processed
                                st.session_state.clarified_parsed = new_parsed
                                st.success(
                                    "✅ Clarification accepted! Proceeding to solve..."
                                )
                                st.rerun()  # Rerun to continue with clarified problem
                    else:
                        st.error(
                            "❌ Please provide clarification text before submitting."
                        )
                        st.stop()

            with col2:
                if st.button("❌ Cancel", type="secondary", key="clarify_cancel"):
                    # Clean up clarification session state
                    if "clarification_submitted" in st.session_state:
                        st.session_state.clarification_submitted = False
                    if "clarified_text" in st.session_state:
                        del st.session_state.clarified_text
                    if "clarified_parsed" in st.session_state:
                        del st.session_state.clarified_parsed
                    st.info("Cancelled. Please enter a new problem.")
                    st.stop()

            # If user hasn't clicked a button yet, stop here
            st.stop()

        trace.log("Parser", f"✅ Topic: {parsed.get('topic', 'unknown')}")

        # ========== STEP 2: ROUTE ==========
        selected_model = st.session_state.get("selected_model")
        topic = route_topic(parsed, selected_model)
        trace.log("Router", f"Routed to topic: {topic}")
        print(f"✅ Routed to: {topic}")

        # ========== STEP 3: RETRIEVE CONTEXT (KB and Web Search in parallel) ==========
        trace.log("Retriever", f"Fetching knowledge for topic: {topic}")
        import concurrent.futures

        # Run KB retrieval and web search in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            kb_future = executor.submit(
                retrieve_context, parsed.get("problem_text"), topic
            )
            web_future = executor.submit(search_web, parsed.get("problem_text"), topic)

            with st.spinner("Retrieving knowledge and searching web..."):
                docs, rag_citations = kb_future.result()
                web_results = web_future.result()

        # Always add web search results (run in parallel, not conditional)
        web_search_formatted = (
            f"Web Search Results:\n{web_results}" if web_results else None
        )
        if web_search_formatted:
            docs.append(web_search_formatted)

        trace.log(
            "Retriever",
            f"✅ Retrieved {len(docs) - (1 if web_search_formatted else 0)} chunk(s) from knowledge base",
        )
        if web_search_formatted:
            trace.log("Web Search", "✅ Web search completed")
        trace.log("Retriever", f"✅ Total {len(docs)} chunks (KB + web)")
        print(
            f"✅ Retrieved {len(docs)} context chunks (KB: {len(docs) - (1 if web_search_formatted else 0)}, Web: {1 if web_search_formatted else 0})"
        )

        # ========== STEP 3.5: CHECK MEMORY ==========
        trace.log("Memory", "Checking for similar past problems...")
        memory_context = memory.get_similar(
            parsed.get("problem_text"), top_k=3, thresh=0.75
        )
        REUSE_SIMILARITY_THRESHOLD = 0.95  # Same/similar question: reuse cached (lowered from 0.99 to catch rephrased questions)
        reused_from_memory = False

        if memory_context:
            sim, past_q, past_sol = memory_context[0]
            print(
                f"🔍 Top similarity: {sim:.4f} (threshold: {REUSE_SIMILARITY_THRESHOLD})"
            )
            print(f"   Current: {parsed.get('problem_text')[:80]}...")
            print(f"   Past:    {past_q[:80]}...")

            if sim >= REUSE_SIMILARITY_THRESHOLD:
                trace.log(
                    "Memory",
                    f"✅ Same/similar question (sim={sim:.4f}); reusing cached solution (saving API calls)",
                )
                print(
                    f"✅ Reusing cached solution for same/similar question (sim={sim:.4f})"
                )
                solution = past_sol
                solution_display = (
                    _solution_to_markdown(past_sol)
                    if past_sol
                    else past_sol or str(past_sol)
                )
                tool_calls = []
                verdict, confidence = "correct", 1.0
                session_mgr.set_agent_state("tool_calls", tool_calls)
                session_mgr.set_agent_state("parsed", parsed)
                session_mgr.set_agent_state("topic", topic)
                reused_from_memory = True
            else:
                trace.log(
                    "Memory",
                    f"✅ Found {len(memory_context)} similar problems (sim={sim:.4f} < {REUSE_SIMILARITY_THRESHOLD}, will solve)",
                )
                print(
                    f"✅ Found {len(memory_context)} similar problems (sim={sim:.4f} < {REUSE_SIMILARITY_THRESHOLD}, solving new)"
                )
        else:
            trace.log("Memory", "No similar problems found in memory")
            print("✅ No similar problems found in memory")

        if not reused_from_memory:
            # ========== STEP 4: SOLVE ==========
            trace.log("Solver", "Solving problem...")
            print("🔧 Calling solver agent...")
            with st.spinner("Solving problem..."):
                solution, tool_calls = solve_problem(
                    parsed,
                    docs,
                    memory_context,
                    session_id=session_mgr.get_session_id(),
                    user_id=session_mgr.get_user_id(),
                    selected_model=selected_model,
                )

            # Check for solver errors
            if isinstance(solution, dict) and solution.get("error"):
                error_info = solution["error"]
                error_type = error_info.get("type", "unknown")
                error_msg = error_info.get("message", "Unknown solver error")

                if error_type == "rate_limit":
                    st.error("❌ **Rate Limit Exceeded**")
                    st.warning(error_msg)
                    st.info(
                        "💡 **Suggestions:**\n"
                        "- Wait a few minutes and try again\n"
                        "- Add a Gemini API key (GEMINI_API_KEY) for automatic fallback\n"
                        "- Check https://console.groq.com/settings/billing for Groq usage\n"
                        "- Try a different Groq model (see https://console.groq.com/docs/models)"
                    )
                    return
                elif error_type == "solver_error":
                    st.error(f"❌ **Solver Error**")
                    st.warning(error_msg)
                    if (
                        "tool_use_failed" in error_msg
                        or "Failed to call a function" in error_msg
                    ):
                        st.info(
                            "💡 The solver encountered an issue with tool calling. This may be due to:\n"
                            "- Groq API limitations\n"
                            "- Ensure you have GEMINI_API_KEY set for fallback\n"
                            "- Try rephrasing your question more simply"
                        )
                    return
                else:
                    st.error(f"❌ Solver error: {error_msg}")
                    return

            solution_display = _solution_to_markdown(solution)
            print(
                f"✅ Solution generated: {solution_display[:100] if solution_display else 'N/A'}..."
            )
            print(f"✅ Tool calls: {len(tool_calls)}")

            session_mgr.set_agent_state("tool_calls", tool_calls)
            session_mgr.set_agent_state("parsed", parsed)
            session_mgr.set_agent_state("topic", topic)

            trace.log("Solver", f"✅ Solution generated ({len(tool_calls)} tool calls)")

            # ========== STEP 5: VERIFY ==========
            trace.log("Verifier", "Verifying solution correctness...")
            with st.spinner("Verifying solution..."):
                verdict, confidence = verify_solution(
                    solution_display or str(solution), parsed, selected_model
                )

            print(f"✅ Verification: {verdict}, confidence: {confidence}")

            trace.log(
                "Verifier", f"✅ Verdict: {verdict}, Confidence: {confidence*100:.1f}%"
            )

            # ========== HITL: Verifier Confidence Check ==========
            # Persist resume data so "Accept" works after rerun (orchestrate not called on button rerun)
            _resume = {
                "parsed": parsed,
                "topic": topic,
                "docs": docs,
                "rag_citations": rag_citations,
                "solution": solution,
                "solution_display": solution_display or str(solution),
                "tool_calls": tool_calls,
                "verdict": verdict,
                "confidence": confidence,
                "memory_context": memory_context,
            }
            st.session_state.verifier_pending_resume = _resume

            if verdict == "incorrect":
                st.error(f"❌ **Verifier Assessment: Solution appears INCORRECT**")
                st.warning(f"Confidence: {confidence*100:.1f}%")
                st.markdown("The verifier identified potential errors. You can:")

                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button(
                        "❌ Reject & Stop",
                        key="verifier_reject",
                        use_container_width=True,
                    ):
                        reset_to_home()
                with col2:
                    if st.button(
                        "⚠️ Accept & show solution",
                        key="verifier_accept_incorrect",
                        use_container_width=True,
                        type="secondary",
                    ):
                        st.session_state.verifier_accepted_resume = dict(_resume)
                        del st.session_state["verifier_pending_resume"]
                        st.rerun()
                with col3:
                    if st.button(
                        "🏠 Home",
                        key="verifier_home_incorrect",
                        use_container_width=True,
                    ):
                        reset_to_home()
                st.stop()

            elif verdict == "uncertain" or confidence < 0.75:
                st.warning(
                    f"⚠️ **Verifier is uncertain** (Verdict: {verdict}, Confidence: {confidence*100:.1f}%)"
                )
                st.caption(
                    "The verifier cannot confirm correctness with high confidence."
                )
                col1, col2 = st.columns([2, 1])
                with col1:
                    if st.button(
                        "✅ Accept & show solution",
                        type="primary",
                        key="verifier_accept_show",
                    ):
                        st.session_state.verifier_accepted_resume = dict(_resume)
                        del st.session_state["verifier_pending_resume"]
                        st.rerun()
                with col2:
                    if st.button(
                        "🏠 Home",
                        key="verifier_home_uncertain_new",
                        use_container_width=True,
                    ):
                        reset_to_home()
                st.info(
                    "💡 Tip: Try rephrasing your question or providing more context."
                )
                st.stop()

        # ========== STEP 6: EXPLAIN ==========
        trace.log("Explainer", "Generating explanation...")
        with st.spinner("Generating explanation..."):
            explanation = explain_solution(
                solution_display or str(solution), docs, parsed, selected_model
            )

        print(f"✅ Explanation generated: {explanation[:100]}...")

        trace.log("Explainer", "✅ Explanation generated")

        # ========== MEMORY: Store Interaction (skip when reused to avoid double-count) ==========
        if not reused_from_memory:
            memory.store_interaction(
                query=parsed.get("problem_text"),
                topic=topic,
                solution=solution_display or str(solution),
                verdict=verdict,
            )
            print("✅ Stored in memory")
        else:
            print("✅ Cached solution shown (not stored again)")

        # ========== DISPLAY RESULTS ==========
        # Persist result view so feedback-button rerun shows same page (not form)
        st.session_state["result_view_state"] = {
            "parsed": parsed,
            "topic": topic,
            "docs": docs,
            "solution_display": solution_display or str(solution),
            "explanation": explanation or "",
            "verdict": verdict,
            "confidence": confidence,
            "memory_context": memory_context or [],
            "rag_citations": rag_citations,
            "tool_calls": session_mgr.get_agent_state("tool_calls") or [],
            "reused_from_memory": reused_from_memory,
        }
        st.markdown("---")
        st.success("✅ Solution Complete!")
        if reused_from_memory:
            st.caption(
                "📌 This question was previously solved; showing cached solution (not counted again)."
            )

        # Pipeline status in main area (so user sees it without opening sidebar)
        with st.expander("🔍 Pipeline execution trace", expanded=False):
            logs = trace.get_logs()
            if logs:
                for step in logs:
                    st.markdown(f"- **{step['agent']}**: {step['message']}")
        st.markdown("---")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📝 Solution")
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")

        solution_display_final = solution_display or str(solution)
        # Solution in blue-tinted container with proper markdown rendering
        st.markdown(
            f'<div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; border-left: 4px solid #2196f3; margin: 10px 0;">'
            f'<div style="color: #1565c0; font-weight: bold; margin-bottom: 8px;">Solution:</div>',
            unsafe_allow_html=True,
        )
        st.markdown(solution_display_final)
        st.markdown("</div>", unsafe_allow_html=True)

        # Only show Explanation if different from Solution
        if explanation and explanation.strip() != solution_display_final.strip():
            st.subheader("💡 Explanation")
            # Explanation in green-tinted container with proper markdown rendering
            st.markdown(
                f'<div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; border-left: 4px solid #4caf50; margin: 10px 0;">'
                f'<div style="color: #2e7d32; font-weight: bold; margin-bottom: 8px;">Explanation:</div>',
                unsafe_allow_html=True,
            )
            st.markdown(explanation)
            st.markdown("</div>", unsafe_allow_html=True)

        # ========== AGENT THOUGHTS & REASONING ==========
        with st.expander("🧠 Agent thoughts & reasoning", expanded=False):
            # Use container with max height for scrollable content
            thoughts_container = st.container()
            with thoughts_container:
                st.markdown("#### Parser Agent")
                st.json(parsed)
                st.markdown("---")

                st.markdown("#### Router Agent")
                st.markdown(f"**Topic:** `{topic}`")
                st.markdown("---")

                st.markdown("#### Solver Agent")
                tool_calls_data = session_mgr.get_agent_state("tool_calls") or []
                if tool_calls_data:
                    st.markdown(
                        f"**Tool calls:** {len(tool_calls_data)} SymPy operation(s)"
                    )
                    for i, tc in enumerate(tool_calls_data, 1):
                        args = tc.get("arguments", {})
                        st.code(
                            f"{i}. {tc.get('tool', 'unknown')}({args})", language="text"
                        )
                else:
                    st.markdown("No tool calls (direct reasoning).")
                st.markdown("---")

                st.markdown("#### Verifier Agent")
                st.markdown(
                    f"**Verdict:** {verdict}  \n**Confidence:** {confidence*100:.1f}%"
                )
                if memory_context:
                    st.markdown("---")
                    st.markdown("#### Memory")
                    st.markdown(
                        f"Used {len(memory_context)} similar past problem(s) as context."
                    )

        # ========== CITATIONS: Always show KB and Web sections ==========
        rag_docs = [doc for doc in docs if not doc.startswith("Web Search Results:")]
        web_docs = [doc for doc in docs if doc.startswith("Web Search Results:")]

        # Always show KB citations if available
        if rag_docs or rag_citations:
            with st.expander("📚 Knowledge base citations", expanded=False):
                _render_kb_citations(rag_citations, rag_docs_fallback=rag_docs)

        # Always show web search citations (run in parallel, not conditional)
        with st.expander("🌐 Web search citations", expanded=False):
            if web_docs:
                st.caption(
                    "Retrieved from DuckDuckGo (searched in parallel with knowledge base)"
                )
                for doc in web_docs:
                    _render_web_search_content(doc)
            else:
                st.caption("No web search results available")

        # ========== SIMILAR PAST PROBLEMS (Always show at bottom) ==========
        st.markdown("---")
        if memory_context:
            st.subheader("🧠 Similar Past Problems")
            st.caption(
                "Retrieved from memory by embedding similarity (learning from past solutions)"
            )
            for i, (sim, past_q, past_sol) in enumerate(memory_context, 1):
                with st.expander(
                    f"📌 Problem {i} — Similarity: {sim*100:.1f}%", expanded=False
                ):
                    st.markdown(f"**Question:** {past_q}")
                    st.markdown("**Solution:**")
                    _render_markdown_block(past_sol, max_len=800)
                if i < len(memory_context):
                    st.divider()
        else:
            st.subheader("🧠 Similar Past Problems")
            st.caption(
                "No similar problems found in memory. This problem will be stored for future reference."
            )

        # ========== FEEDBACK (only when we stored a new interaction) ==========
        if not reused_from_memory:
            st.markdown("---")
            if st.session_state.get("feedback_success_msg"):
                st.success(st.session_state["feedback_success_msg"])
                st.session_state.pop("feedback_success_msg", None)
        st.subheader("✅ Feedback")
        st.caption("Your feedback helps improve the system.")
        show_fb = st.session_state.get("show_feedback", False)
        if not show_fb:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button(
                    "👍 Correct", key="feedback_correct", use_container_width=True
                ):
                    memory.feedback(True)
                    st.session_state["feedback_success_msg"] = "Thanks for confirming!"
                    reset_to_home()
            with col2:
                if st.button(
                    "👎 Incorrect", key="feedback_incorrect", use_container_width=True
                ):
                    st.session_state["show_feedback"] = True
                    st.rerun()
            with col3:
                if st.button("🏠 Home", key="feedback_home", use_container_width=True):
                    reset_to_home()
        else:
            comment = st.text_area("What was wrong? (optional)", key="feedback_comment")
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("Submit Feedback", key="feedback_submit"):
                    memory.feedback(False, comment)
                    st.session_state["feedback_success_msg"] = "Feedback recorded!"
                    reset_to_home()
            with c2:
                if st.button("Cancel", key="feedback_cancel"):
                    st.session_state["show_feedback"] = False
                    st.rerun()
            with c3:
                if st.button(
                    "🏠 Home", key="feedback_home_comment", use_container_width=True
                ):
                    reset_to_home()

    except Exception as e:
        st.error(f"❌ Error in orchestration: {str(e)}")
        print(f"❌ ORCHESTRATION ERROR: {e}")
        import traceback

        traceback.print_exc()
        st.code(traceback.format_exc())
