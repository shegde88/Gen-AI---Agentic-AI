"""
FIFA World Cup 2026 Fan Intelligence — Streamlit Chat App

Run:
    uv run streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from wc_rag.chain import ask
from wc_rag.indexing import load_vector_store
from wc_rag.retriever import build_dense_retriever

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FIFA World Cup 2026 — Fan Intelligence",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — FIFA 2026 dark theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  /* ── Global ── */
  [data-testid="stAppViewContainer"] {
      background: #0A0E1A;
      color: #E8EAF0;
  }
  [data-testid="stHeader"] { background: transparent; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background: #131929 !important;
      border-right: 1px solid #1E2A45;
  }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #D4AF37; }

  /* ── Hero header ── */
  .wc-hero {
      background: linear-gradient(135deg, #003DA5 0%, #0A0E1A 60%, #C8102E 100%);
      border-radius: 16px;
      padding: 28px 32px 22px;
      margin-bottom: 24px;
      border: 1px solid #1E2A45;
      position: relative;
      overflow: hidden;
  }
  .wc-hero::before {
      content: "⚽";
      position: absolute;
      right: 24px;
      top: 50%;
      transform: translateY(-50%);
      font-size: 5rem;
      opacity: 0.12;
  }
  .wc-hero h1 {
      font-size: 1.8rem;
      font-weight: 800;
      color: #FFFFFF;
      margin: 0 0 6px;
      letter-spacing: 0.5px;
  }
  .wc-hero .subtitle {
      color: #D4AF37;
      font-size: 0.88rem;
      font-weight: 500;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      margin-bottom: 8px;
  }
  .wc-hero .desc {
      color: #8892A4;
      font-size: 0.92rem;
      max-width: 600px;
  }

  /* ── Stat badges in hero ── */
  .badge-row { display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap; }
  .badge {
      background: rgba(255,255,255,0.07);
      border: 1px solid rgba(212,175,55,0.3);
      border-radius: 20px;
      padding: 4px 14px;
      font-size: 0.8rem;
      color: #D4AF37;
      white-space: nowrap;
  }

  /* Style the suggestion buttons to look like chips */
  div[data-testid="stButton"] button {
      background: #131929 !important;
      color: #E8EAF0 !important;
      border: 1px solid #1E2A45 !important;
      border-radius: 10px !important;
      font-size: 0.88rem !important;
      text-align: left !important;
      padding: 10px 16px !important;
      transition: border-color 0.2s, background 0.2s !important;
  }
  div[data-testid="stButton"] button:hover {
      border-color: #D4AF37 !important;
      background: #1A2235 !important;
      color: #FFFFFF !important;
  }

  /* ── Chat messages ── */
  [data-testid="stChatMessage"] {
      background: #131929 !important;
      border: 1px solid #1E2A45 !important;
      border-radius: 12px !important;
      margin-bottom: 8px !important;
  }

  /* ── Chat input ── */
  [data-testid="stChatInput"] textarea {
      background: #131929 !important;
      border: 1px solid #1E2A45 !important;
      border-radius: 12px !important;
      color: #E8EAF0 !important;
  }
  [data-testid="stChatInput"] textarea:focus {
      border-color: #D4AF37 !important;
      box-shadow: 0 0 0 2px rgba(212,175,55,0.2) !important;
  }

  /* ── Divider ── */
  hr { border-color: #1E2A45 !important; }

  /* ── Success / warning / error ── */
  [data-testid="stAlert"] { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="wc-hero">
  <div class="subtitle">🏆 FIFA World Cup 2026 · Fan Intelligence</div>
  <h1>Your World Cup Expert</h1>
  <div class="desc">
    Ask anything about the 2026 tournament — teams, players, venues, match history,
    and records from 1930 to today. Every answer is grounded in sources and cited.
  </div>
  <div class="badge-row">
    <span class="badge">⚽ 48 Teams</span>
    <span class="badge">🌎 USA · Canada · Mexico</span>
    <span class="badge">📚 85 Wikipedia Sources</span>
    <span class="badge">📊 Match Data 1930–2022</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown("### 💡 Sample Questions")
    st.caption("Click any question to ask it instantly")

    SUGGESTED_QUESTIONS = [
        "🏆 Which three countries are hosting the 2026 World Cup?",
        "⚽ How many teams are playing in the 2026 World Cup?",
        "🥅 How many World Cup titles does Brazil have?",
        "🌟 How many World Cup goals has Lionel Messi scored?",
        "📍 Where is the 2026 World Cup final being held?",
        "🇦🇷 Who won the 2022 FIFA World Cup?",
        "📊 Who is the all-time top scorer in World Cup history?",
        "🇲🇦 What was Morocco's best-ever World Cup result?",
    ]
    for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
        if st.button(suggestion, use_container_width=True, key=f"suggest_{i}"):
            st.session_state["_pending_question"] = suggestion
            st.rerun()

    st.divider()
    st.caption("Powered by **Nebius** · **Pinecone** · **LangGraph**")

# ---------------------------------------------------------------------------
# Load retriever (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_retriever():
    vs = load_vector_store()
    return build_dense_retriever(vs)

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Chat input + question dispatch
# ---------------------------------------------------------------------------

typed_question = st.chat_input("Ask about the 2026 World Cup...")
question = st.session_state.pop("_pending_question", None) or typed_question

if question:
    clean_question = question.lstrip("🏆⚽🥅🌟📍🇦🇷📊🇲🇦 ")

    st.session_state.messages.append({"role": "user", "content": clean_question})
    with st.chat_message("user"):
        st.markdown(clean_question)

    try:
        with st.spinner("Searching knowledge base..."):
            retriever = get_retriever()
            result = ask(
                question=clean_question,
                retriever=retriever,
                chat_history=st.session_state.chat_history,
            )

        answer_body = result["answer"]
        routed_to_answer = result.get("routing_decision") == "answer"
        has_docs = bool(result.get("documents"))
        citations = result["citations"] if (routed_to_answer and has_docs) else []

        if citations:
            citation_block = "\n\n---\n**Sources:** " + " · ".join(f"`{c}`" for c in citations)
        else:
            citation_block = ""

        full_response = answer_body + citation_block

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.chat_history.append((clean_question, answer_body))

        with st.chat_message("assistant"):
            st.markdown(full_response)

    except Exception as exc:
        err_msg = f"Something went wrong: {exc}"
        st.session_state.messages.append({"role": "assistant", "content": err_msg})
        with st.chat_message("assistant"):
            st.error(err_msg)
