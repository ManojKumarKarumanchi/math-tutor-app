# Complete System Flow 

## 1. Model Selection Flow

```
┌─────────────────────────────────────────┐
│          Streamlit App Starts           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│   Check for API Keys in .env            │
│   - OPENROUTER_API_KEY                  │
│   - GROQ_API_KEY                        │
│   - GEMINI_API_KEY                      │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
┌──────────────┐   ┌──────────────────┐
│ Keys Found   │   │ No Keys Found    │
└──────┬───────┘   └──────┬───────────┘
       │                  │
       ▼                  ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│ Show Provider Dropdown   │   │ Show Error Message:      │
│ (OpenRouter/Groq/Gemini) │   │ "No API keys configured" │
│                          │   │ + Setup Instructions     │
└──────────┬───────────────┘   └──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Show Model Dropdown      │
│ (filtered by provider)   │
│ e.g., Qwen3 235B, Llama  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ st.session_state         │
│ .selected_model = model  │
└──────────────────────────┘
```

## 2. Input Processing Flow

```
┌─────────────────────────────────────────┐
│         User Provides Input             │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌──────┐   ┌──────────┐   ┌────────┐
│ Text │   │  Image   │   │ Audio  │
└──┬───┘   └────┬─────┘   └───┬────┘
   │            │             │
   │            ▼             ▼
   │      ┌─────────┐   ┌──────────┐
   │      │ EasyOCR │   │  Groq    │
   │      │ (Fixed) │   │ Whisper  │
   │      │         │   │ (Fixed)  │
   │      └────┬────┘   └────┬─────┘
   │           │             │
   │           ▼             ▼
   │      ┌────────────┐   ┌───────────┐
   │      │ Conf < 0.6 │   │ Conf<0.6  │
   │      │ HITL Loop  │   │ HITL Loop │
   │      └────┬───────┘   └────┬──────┘
   │           │                │
   └───────────┴────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │ Combined Text  │
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │ Check Model    │
        │ Selected?      │
        └────────┬───────┘
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
   ┌─────────┐      ┌──────────────┐
   │  YES    │      │  NO → Error  │
   │ Proceed │      │  "Select a   │
   └────┬────┘      │   model"     │
        │           └──────────────┘
        ▼
   ┌────────────┐
   │ Orchestrate│
   └────────────┘
```

## 3. Agent Pipeline Flow (with Model Selection)

```
┌─────────────────────────────────────────────────┐
│  orchestrate(processed, selected_model, ...)    │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  1. GUARDRAIL AGENT (selected_model)            │
│     ├─ Check safety & scope                     │
│     └─ Result: approved/rejected                │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  2. PARSER AGENT (selected_model)               │
│     ├─ Extract structure: topic, variables, etc │
│     ├─ Detect ambiguity                         │
│     └─ Trigger HITL if needs_clarification=True │
└────────────────────┬────────────────────────────┘
                     │
            ┌────────┴────────┐
            │                 │
            ▼                 ▼
    ┌───────────────┐   ┌───────────────┐
    │  Ambiguous    │   │  Clear        │
    └───────┬───────┘   └───────┬───────┘
            │                   │
            ▼                   │
    ┌───────────────┐           │
    │  HITL: User   │           │
    │  Clarifies    │           │
    └───────┬───────┘           │
            │                   │
            └───────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  3. ROUTER AGENT                                │
│     └─ Classify topic (algebra, calculus, etc)  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  4. RAG RETRIEVAL                               │
│     ├─ LanceDB vector search                    │
│     ├─ Check sufficiency                        │
│     └─ Optional: DuckDuckGo web search          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  5. MEMORY CHECK (Similarity Search)            │
│     └─ If similarity ≥ 0.99: Reuse cached       │
└────────────────────┬────────────────────────────┘
                     │
            ┌────────┴────────┐
            │                 │
            ▼                 ▼
    ┌───────────────┐   ┌───────────────────────┐
    │  Cached       │   │  Not Cached           │
    │  (Reuse)      │   └───────┬───────────────┘
    └───────┬───────┘           │
            │                   ▼
            │           ┌───────────────────────┐
            │           │ 6. SOLVER AGENT       │
            │           │    (selected_model)   │
            │           │    ├─ RAG context     │
            │           │    ├─ Memory context  │
            │           │    ├─ SymPy tools     │
            │           │    └─ Generate solution│
            │           └───────┬───────────────┘
            │                   │
            │                   ▼
            │           ┌───────────────────────┐
            │           │ Error Check           │
            │           │ ├─ Rate limit?        │
            │           │ ├─ Solver error?      │
            │           │ └─ Show UI error      │
            │           └───────┬───────────────┘
            │                   │
            │                   ▼
            │           ┌───────────────────────┐
            │           │ 7. VERIFIER AGENT     │
            │           │    (selected_model)   │
            │           │    └─ verdict,        │
            │           │       confidence      │
            │           └───────┬───────────────┘
            │                   │
            │          ┌────────┴────────┐
            │          │                 │
            │          ▼                 ▼
            │    ┌───────────┐   ┌──────────────┐
            │    │ Conf<0.7  │   │  Conf≥0.7    │
            │    │ HITL:     │   └──────┬───────┘
            │    │ Accept/   │          │
            │    │ Reject    │          │
            │    └─────┬─────┘          │
            │          │                │
            │          └────────────────┘
            │                   │
            └───────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  8. EXPLAINER AGENT (selected_model)            │
│     └─ Generate student-friendly explanation    │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  9. MEMORY STORAGE (if not cached)              │
│     ├─ Store problem + solution + embeddings    │
│     ├─ Track last_entry_id                      │
│     └─ Skip if reused_from_memory=True          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  10. DISPLAY RESULTS                            │
│      ├─ Solution with confidence                │
│      ├─ Explanation                             │
│      ├─ Pipeline trace                          │
│      ├─ KB citations                            │
│      ├─ Web citations (if used)                 │
│      └─ Similar past problems (if any)          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  11. FEEDBACK SECTION (if not cached)           │
│      ├─ 👍 Correct button                       │
│      └─ 👎 Incorrect button                     │
└────────────────────┬────────────────────────────┘
                     │
            ┌────────┴─────────┐
            │                  │
            ▼                  ▼
    ┌───────────────┐   ┌──────────────────┐
    │  👍 Clicked   │   │  👎 Clicked      │
    └───────┬───────┘   └──────┬───────────┘
            │                  │
            ▼                  ▼
    ┌───────────────┐   ┌──────────────────┐
    │ feedback(True)│   │ Show Comment Box │
    │ Update DB:    │   └──────┬───────────┘
    │ feedback=1    │          │
    └───────┬───────┘          ▼
            │           ┌──────────────────┐
            │           │ Submit Feedback  │
            │           │ feedback(False,  │
            │           │  comment)        │
            │           │ Update DB:       │
            │           │ feedback=0       │
            │           └──────┬───────────┘
            │                  │
            └──────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  12. UPDATE STATS IN SIDEBAR                    │
│      ├─ Total Problems                          │
│      ├─ 👍 Positive Feedback                    │
│      └─ 👎 Negative Feedback                    │
└─────────────────────────────────────────────────┘
```

## 4. Error Handling Flow

```
┌─────────────────────────────────────────┐
│          Error Occurs                   │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌──────────┐ ┌─────────┐ ┌──────────┐
│ No API   │ │  Rate   │ │  Agent   │
│  Keys    │ │  Limit  │ │  Error   │
└────┬─────┘ └────┬────┘ └────┬─────┘
     │            │            │
     ▼            ▼            ▼
┌──────────────────────────────────────┐
│ Display Error Message:               │
│ ❌ [Error Title]                     │
│ ⚠️  [Error Details]                  │
│ 💡 [Suggestions]                     │
│    - Action 1                        │
│    - Action 2                        │
│    - Action 3                        │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ st.stop() - Halt Pipeline            │
│ (User can fix and retry)             │
└──────────────────────────────────────┘
```

## 5. Feedback & Memory Sync Flow

```
┌─────────────────────────────────────────┐
│  Solution Displayed (not cached)        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  memory.store_interaction()             │
│  ├─ Insert into DB: problems table      │
│  ├─ Generate embedding                  │
│  ├─ Store: query, solution, topic, etc  │
│  └─ Save last_entry_id                  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Show Feedback Buttons                  │
│  ├─ 👍 Correct                          │
│  └─ 👎 Incorrect                        │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
┌──────────────┐   ┌───────────────────┐
│ User clicks  │   │ User clicks       │
│ 👍 Correct   │   │ 👎 Incorrect      │
└──────┬───────┘   └────────┬──────────┘
       │                    │
       ▼                    ▼
┌──────────────────┐   ┌───────────────────┐
│ memory.feedback  │   │ Show comment      │
│ (True)           │   │ text area         │
│                  │   └────────┬──────────┘
│ SQL:             │            │
│ UPDATE problems  │            ▼
│ SET feedback=1   │   ┌───────────────────┐
│ WHERE id=        │   │ User submits      │
│ last_entry_id    │   │ (with comment)    │
└──────┬───────────┘   └────────┬──────────┘
       │                        │
       │                        ▼
       │               ┌───────────────────┐
       │               │ memory.feedback   │
       │               │ (False, comment)  │
       │               │                   │
       │               │ SQL:              │
       │               │ UPDATE problems   │
       │               │ SET feedback=0,   │
       │               │ comment='...'     │
       │               │ WHERE id=         │
       │               │ last_entry_id     │
       │               └────────┬──────────┘
       │                        │
       └────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ Show Success Message                    │
│ ✅ "Thanks for confirming!" or          │
│ ✅ "Feedback recorded!"                 │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ st.rerun()                              │
│ ├─ Clear result_view_state              │
│ ├─ Reset show_feedback flag             │
│ └─ Trigger sidebar stats refresh        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ Sidebar Stats Update                    │
│                                         │
│ memory.get_stats():                     │
│ SELECT COUNT(*) FROM problems           │
│   WHERE feedback=1  → Positive          │
│   WHERE feedback=0  → Negative          │
│   Total             → Total             │
│                                         │
│ Display:                                │
│ ├─ Total Problems: X                    │
│ ├─ 👍 Positive: Y                       │
│ └─ 👎 Negative: Z                       │
└─────────────────────────────────────────┘
```