"""
Streamlit UI for the CV QA Agent.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV QA Agent",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .user-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 85%;
        margin-left: auto;
    }
    .agent-bubble {
        background: #1e2330;
        color: #e0e0e0;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 90%;
        border-left: 3px solid #667eea;
    }
    .turn-badge {
        font-size: 0.7rem;
        color: #555;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state init ────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn" not in st.session_state:
    st.session_state.turn = 0
if "output_entries" not in st.session_state:
    # Each entry: {"timestamp": str, "question": str, "answer": str}
    st.session_state.output_entries = []
if "session_started" not in st.session_state:
    st.session_state.session_started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Helper: build output.txt content from session_state ──────────────────────

def _build_output_text() -> str:
    """Generate the full session output text from current session_state entries."""
    entries = st.session_state.output_entries
    lines = [
        "=" * 70,
        "CV QA AGENT — SESSION OUTPUT",
        f"Session started : {st.session_state.session_started}",
        f"Last updated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total questions : {len(entries)}",
        "=" * 70,
        "",
    ]
    for i, e in enumerate(entries, start=1):
        lines += [
            f"[{e['timestamp']}] Question {i}:",
            f"  Q: {e['question']}",
            f"  A: {e['answer']}",
            "",
        ]
    return "\n".join(lines)


# ── Agent initialisation (cached across reruns) ───────────────────────────────

@st.cache_resource(show_spinner="Initialising CV Agent… (indexing may take a moment)")
def get_agent(cv_paths_tuple: tuple[str, ...] | None = None):
    """Load and initialise the CVAgent singleton for the given CV file(s)."""
    from cv_agent import CVAgent
    paths = [Path(p) for p in cv_paths_tuple] if cv_paths_tuple else None
    return CVAgent(cv_paths=paths)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 Upload CVs")
    uploaded_files = st.file_uploader(
        "Upload candidate CVs (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload one or multiple CVs to index, query, and compare candidates.",
    )

    saved_paths: list[str] = []
    if uploaded_files:
        for uf in uploaded_files:
            save_path = Path("data") / uf.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(str(save_path))

        current_set = tuple(sorted(saved_paths))
        if "loaded_cv_set" not in st.session_state or st.session_state.loaded_cv_set != current_set:
            st.session_state.loaded_cv_set = current_set
            st.session_state.messages = []
            st.session_state.turn = 0
            st.session_state.output_entries = []
            st.session_state.session_started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.cache_resource.clear()
            st.success(f"Indexed **{len(saved_paths)}** CV(s)!")
            st.rerun()

    # Fall back to data/ directory files if nothing uploaded via UI
    if not saved_paths:
        data_files = list(Path("data").glob("*.pdf")) + list(Path("data").glob("*.docx"))
        if data_files:
            saved_paths = [str(p) for p in data_files]

    if saved_paths:
        st.caption(f"**{len(saved_paths)} Active CV(s):**")
        for p in saved_paths:
            st.markdown(f"- 📄 `{Path(p).name}`")

    st.divider()

    # ── Output / Export ───────────────────────────────────────────────────────
    st.markdown("## 💾 Export Session")

    n = len(st.session_state.output_entries)
    if n > 0:
        st.caption(f"**{n} question(s)** in this session.")
        output_text = _build_output_text()

        # Download button — always reflects the CURRENT session entries
        st.download_button(
            label="⬇️ Download output.txt",
            data=output_text.encode("utf-8"),
            file_name="output.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.caption("No questions yet — ask something to enable export.")

    st.divider()

    # ── Session controls ──────────────────────────────────────────────────────
    st.markdown("## ⚙️ Session Controls")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.turn = 0
        st.session_state.output_entries = []
        st.session_state.session_started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            cv_tuple = tuple(saved_paths) if saved_paths else None
            get_agent(cv_tuple).clear_history()
        except Exception:
            pass
        st.rerun()

    st.divider()

    st.markdown("### About")
    st.info(
        "This agent answers questions strictly from candidate CVs. "
        "Supports single-candidate lookup and cross-candidate comparisons."
    )

    from cv_agent.config import settings as cfg
    st.markdown(f"""
**Configuration**
- Provider: `{cfg.llm_provider}`
- Model: `{cfg.llm_model}`
- Chunk size: `{cfg.chunk_size}`
- Top-k: `{cfg.retrieval_top_k}`
- Active files: `{len(saved_paths)}`
""")


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">📄 CV Question Answering Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by LangGraph · Google Gemini · FAISS</div>', unsafe_allow_html=True)

# Load agent
try:
    cv_tuple = tuple(saved_paths) if saved_paths else None
    agent = get_agent(cv_tuple)
except FileNotFoundError as exc:
    st.error(f"**CV file not found:** {exc}")
    st.stop()
except Exception as exc:
    st.error(f"**Agent initialisation failed:** {exc}")
    st.stop()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        st.markdown(
            f'<div class="turn-badge">You</div>'
            f'<div class="user-bubble">{content}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="turn-badge">Agent (turn {msg.get("turn", "")})</div>'
            f'<div class="agent-bubble">{content}</div>',
            unsafe_allow_html=True,
        )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the CV…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f'<div class="turn-badge">You</div>'
        f'<div class="user-bubble">{prompt}</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Searching CV and reasoning…"):
        t0 = time.perf_counter()
        answer = agent.ask(prompt)
        elapsed = time.perf_counter() - t0

    st.session_state.turn += 1
    st.session_state.messages.append(
        {"role": "agent", "content": answer, "turn": st.session_state.turn}
    )

    # Record into session_state so the download button always has latest data
    st.session_state.output_entries.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": prompt,
        "answer": answer,
    })

    st.markdown(
        f'<div class="turn-badge">Agent (turn {st.session_state.turn}) '
        f'· {elapsed:.1f}s</div>'
        f'<div class="agent-bubble">{answer}</div>',
        unsafe_allow_html=True,
    )
