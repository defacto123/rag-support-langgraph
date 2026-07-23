"""Streamlit chat UI for the RAG document assistant.

Talks to the FastAPI backend over HTTP:
  POST /upload -> ingest a document
  POST /chat   -> ask a question (keeps a per-session thread_id for memory)

Run with:
  streamlit run app/ui/streamlit_app.py
"""

import base64
import os
import uuid
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8010")

# Feature flags (default = original behaviour: upload enabled, generic title).
# A pre-indexed deployment (e.g. MobiSystems KB) sets DISABLE_UPLOAD=true and
# a custom title, turning the app into a chat-only assistant.
DISABLE_UPLOAD = os.getenv("DISABLE_UPLOAD", "false").lower() == "true"
APP_TITLE = os.getenv("APP_TITLE", "Your Document Assistant")
APP_SUBTITLE = os.getenv(
    "APP_SUBTITLE", "Ask a question — I answer only from your documents."
)
# Source metadata is hidden from users by default; flip this on to show the
# Sources expander when debugging retrieval.
SHOW_SOURCES = os.getenv("SHOW_SOURCES", "false").lower() == "true"

# --- Brand assets (bundled in the image so they deploy with the app) ---
_ASSETS = Path(__file__).parent / "assets"
_ICON_PATH = _ASSETS / "mobi_icon.jpg"
_LOGO_PATH = _ASSETS / "mobi_logo.png"


def _data_uri(path: Path, mime: str) -> str:
    """Base64 data URI so an image can be embedded directly in custom HTML."""
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except OSError:
        return ""


_ICON_URI = _data_uri(_ICON_PATH, "image/jpeg")
_LOGO_URI = _data_uri(_LOGO_PATH, "image/png")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=str(_ICON_PATH) if _ICON_PATH.exists() else "💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- MobiSystems theme (blue-on-white, clean & modern, matches mobisystems.com) ---
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      :root {
        --bg:          #F5F8FC;
        --surface:     #FFFFFF;
        --primary:     #038FF3;
        --primary-2:   #00AFFF;
        --primary-dark:#226ECF;
        --heading:     #243278;
        --text:        #313131;
        --muted:       #5B5B5B;
        --border:      #E6E6E6;
      }

      .stApp { background: var(--bg); }
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }

      /* Hide Streamlit chrome for a cleaner look */
      #MainMenu, footer, header { visibility: hidden; }
      .block-container { padding-top: 2.5rem; padding-bottom: 7rem; max-width: 760px; }

      /* Mobi widget top bar (logo + name + subtitle) */
      .mobi-topbar {
        display: flex; align-items: center; gap: 0.7rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 2px 12px rgba(3, 143, 243, 0.06);
      }
      .mobi-logo {
        width: 36px; height: 36px; flex: 0 0 36px;
        border-radius: 9px;
        background: linear-gradient(135deg, #F0483E, #C40E0E);
        color: #fff; font-weight: 800; font-size: 1.15rem;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 2px 6px rgba(196, 14, 14, 0.35);
      }
      /* Actual brand icon (red "M") used in the header top bar */
      .mobi-logo-img {
        width: 38px; height: 38px; flex: 0 0 38px;
        border-radius: 9px; object-fit: cover;
        box-shadow: 0 2px 6px rgba(196, 14, 14, 0.35);
      }
      .mobi-titles { line-height: 1.2; }
      .mobi-name { font-weight: 700; color: var(--heading); font-size: 1rem; }
      /* MobiSystems wordmark image */
      .mobi-wordmark { height: 20px; display: block; margin-bottom: 2px; }
      .mobi-sub  { color: var(--muted); font-size: 0.8rem; margin-top: 1px; }

      /* Meta line under assistant replies (e.g. "Mobi · AI Agent · Just now") */
      .mobi-meta {
        color: var(--muted); font-size: 0.75rem;
        margin: -0.35rem 0 0.7rem 0.2rem;
      }

      /* "Agent is typing…" indicator — three bouncing dots (replaces the
         plain "Thinking…" spinner while we wait for the backend reply) */
      .typing-indicator {
        display: flex; align-items: center; gap: 5px;
        height: 1.4rem;              /* give the dots a line box to centre in */
      }
      .typing-indicator span {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--muted); display: inline-block;
        vertical-align: middle;
        animation: mobi-typing 1.2s infinite ease-in-out both;
      }
      .typing-indicator span:nth-child(1) { animation-delay: -0.24s; }
      .typing-indicator span:nth-child(2) { animation-delay: -0.12s; }
      .typing-indicator span:nth-child(3) { animation-delay: 0s; }
      @keyframes mobi-typing {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40%           { transform: scale(1);   opacity: 1;   }
      }

      /* Chat bubbles — hide avatars, use aligned coloured bubbles like the widget */
      [data-testid="stChatMessageAvatarUser"],
      [data-testid="stChatMessageAvatarAssistant"] { display: none !important; }

      [data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        box-shadow: none;
        padding: 0;
        margin-bottom: 0.5rem;
      }
      /* the inner content wrapper becomes the actual bubble */
      [data-testid="stChatMessage"] > div:last-child {
        border-radius: 16px;
        padding: 0.7rem 0.95rem;
        max-width: 82%;
        width: fit-content;
      }
      /* Assistant: light-grey bubble, left aligned */
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div:last-child {
        background: #F1F3F4;
        color: var(--text);
        margin-right: auto;
      }
      /* User: brand-blue bubble, right aligned, white text */
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child {
        background: var(--primary);
        margin-left: auto;
      }
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) li,
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span {
        color: #ffffff !important;
      }

      /* Chat input bar — lift it and make it clearly a text field */
      [data-testid="stBottomBlockContainer"], [data-testid="stChatInput"] {
        background: var(--bg) !important;
      }
      [data-testid="stChatInput"] {
        padding-bottom: 1.25rem;
      }
      [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
        font-size: 1rem !important;
        border-radius: 14px !important;
        border: 1.5px solid var(--primary-2) !important;
        background: var(--surface) !important;
        box-shadow: 0 4px 16px rgba(3, 143, 243, 0.10) !important;
        /* breathing room so text/placeholder isn't glued to the frame */
        padding: 0.6rem 0.95rem !important;
        line-height: 1.4 !important;
      }
      [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important;
        opacity: 1 !important;
      }
      [data-testid="stChatInput"] textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(0, 175, 255, 0.30) !important;
      }
      /* Send button — circular blue like the Mobi widget */
      [data-testid="stChatInputSubmitButton"] {
        background: var(--primary) !important;
        color: #ffffff !important;
        border-radius: 50% !important;
        width: 2.1rem !important; height: 2.1rem !important;
      }
      [data-testid="stChatInputSubmitButton"]:disabled {
        background: #C9D3DE !important;
        color: #ffffff !important;
      }
      [data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {
        background: var(--primary-dark) !important;
      }

      /* Buttons — keep text readable in every state (incl. inner span/p) */
      .stButton button, .stDownloadButton button {
        background-color: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 12px; padding: 0.55rem 1.1rem; font-weight: 600;
        transition: all 0.15s ease;
      }
      .stButton button p, .stButton button span, .stButton button div,
      .stDownloadButton button p, .stDownloadButton button span {
        color: #ffffff !important;
      }
      .stButton button:hover, .stDownloadButton button:hover,
      .stButton button:focus, .stButton button:active,
      .stButton button:focus:not(:active) {
        background-color: var(--primary-dark) !important;
        border-color: var(--primary-dark) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(3, 143, 243, 0.30) !important;
      }
      .stButton button:hover p, .stButton button:hover span, .stButton button:hover div,
      .stButton button:focus p, .stButton button:focus span,
      .stButton button:active p, .stButton button:active span {
        color: #ffffff !important;
      }

      /* Sidebar */
      [data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
      [data-testid="stSidebar"] h2 { font-size: 1.05rem; font-weight: 700; color: var(--heading); }

      /* Source chips */
      .source-card {
        background: #EEF6FF; border: 1px solid var(--border);
        border-radius: 12px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;
        font-size: 0.85rem; color: var(--text);
      }
      .source-card .src-name { font-weight: 600; color: var(--primary); }
      .source-card .src-preview { color: var(--muted); margin-top: 0.2rem; }

      /* File uploader */
      [data-testid="stFileUploaderDropzone"] {
        background: #EEF6FF; border: 1.5px dashed var(--primary-2);
        border-radius: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session state ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: document upload (optional) + controls ---
with st.sidebar:
    if _LOGO_URI:
        st.markdown(
            f'<img src="{_LOGO_URI}" alt="MobiSystems" '
            f'style="height:26px;margin:0.1rem 0 1rem;"/>',
            unsafe_allow_html=True,
        )
    if not DISABLE_UPLOAD:
        st.markdown("## 📎 Documents")
        st.caption("Upload a file to expand the assistant's knowledge.")

        uploaded = st.file_uploader(
            "PDF, TXT, MD or DOCX",
            type=["pdf", "txt", "md", "docx"],
            label_visibility="collapsed",
        )
        if uploaded is not None and st.button(
            "Index document", use_container_width=True
        ):
            with st.spinner("Processing document…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded.name, uploaded.getvalue())},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.toast(f"{data['file']} indexed successfully", icon="✅")
                    st.success(
                        f"**Done! The document has been saved and indexed.**\n\n"
                        f"- File: `{data['file']}`\n"
                        f"- Chunks: **{data['chunks']}**\n"
                        f"- Chunking type: `{data['doc_type']}`\n\n"
                        f"You can now ask questions about it."
                    )
                except requests.RequestException as exc:
                    st.error(f"Upload error: {exc}")

        st.divider()

    if st.button("New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# --- Header: Mobi widget top bar (uses brand assets, falls back to text) ---
_icon_html = (
    f'<img class="mobi-logo-img" src="{_ICON_URI}" alt="Mobi"/>'
    if _ICON_URI
    else '<div class="mobi-logo">M</div>'
)
_name_html = (
    f'<img class="mobi-wordmark" src="{_LOGO_URI}" alt="MobiSystems"/>'
    if _LOGO_URI
    else '<div class="mobi-name">Mobi</div>'
)
st.markdown(
    f"""
    <div class="mobi-topbar">
      {_icon_html}
      <div class="mobi-titles">
        {_name_html}
        <div class="mobi-sub">AI Agent • The team can also help</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _assistant_meta() -> None:
    """Small identity line shown under each assistant reply (widget style)."""
    st.markdown(
        '<div class="mobi-meta">Mobi &bull; AI Agent &bull; Just now</div>',
        unsafe_allow_html=True,
    )


# Animated "typing" bubble shown while the backend is composing the reply.
_TYPING_HTML = (
    '<div class="typing-indicator"><span></span><span></span><span></span></div>'
)


def _render_sources(sources: list) -> None:
    with st.expander("Sources"):
        for s in sources:
            st.markdown(
                f"""
                <div class="source-card">
                  <span class="src-name">[{s['index']}] {s['source']}</span>
                  <div class="src-preview">{s['preview']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --- Welcome greeting (display-only; not part of the model context) ---
with st.chat_message("assistant"):
    st.markdown(
        "Hi there! You're speaking with **Mobi AI Agent**.\n\n"
        "I can help you find answers and get support for:\n\n"
        "- MobiOffice\n- MobiPDF\n- MobiDrive\n\n"
        "How can I assist you today?"
    )
_assistant_meta()

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if SHOW_SOURCES and msg.get("sources"):
            _render_sources(msg["sources"])
    if msg["role"] == "assistant":
        _assistant_meta()

# --- Chat input ---
if prompt := st.chat_input("Message…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Show an animated "typing…" bubble, then swap it for the real answer.
        placeholder = st.empty()
        placeholder.markdown(_TYPING_HTML, unsafe_allow_html=True)
        try:
            resp = requests.post(
                f"{API_URL}/chat",
                json={"question": prompt, "thread_id": st.session_state.thread_id},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["answer"]
            sources = data.get("sources", [])
        except requests.RequestException as exc:
            answer = f"Something went wrong connecting to the backend: {exc}"
            sources = []

        placeholder.markdown(answer)
        if SHOW_SOURCES and sources:
            _render_sources(sources)
    _assistant_meta()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
