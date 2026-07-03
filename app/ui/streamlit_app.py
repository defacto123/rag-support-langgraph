"""Streamlit chat UI for the RAG document assistant.

Talks to the FastAPI backend over HTTP:
  POST /upload -> ingest a document
  POST /chat   -> ask a question (keeps a per-session thread_id for memory)

Run with:
  streamlit run app/ui/streamlit_app.py
"""

import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8010")

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Warm, minimalist theme (soft lines, warm palette) ---
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

      :root {
        --bg:        #FBF7F2;
        --surface:   #FFFFFF;
        --primary:   #E07A5F;
        --primary-2: #F2A488;
        --text:      #3D3A36;
        --muted:     #9C948B;
        --border:    #EFE7DD;
      }

      .stApp { background: var(--bg); }
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }

      /* Hide Streamlit chrome for a cleaner look */
      #MainMenu, footer, header { visibility: hidden; }
      .block-container { padding-top: 2.5rem; padding-bottom: 7rem; max-width: 760px; }

      /* Hero header */
      .hero { text-align: center; margin-bottom: 1.75rem; }
      .hero h1 {
        font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;
        margin: 0; color: var(--text);
      }
      .hero p { color: var(--muted); margin: 0.35rem 0 0; font-size: 0.95rem; }

      /* Chat bubbles */
      [data-testid="stChatMessage"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.35rem 0.35rem;
        box-shadow: 0 2px 12px rgba(224, 122, 95, 0.06);
        margin-bottom: 0.6rem;
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
        border-radius: 16px !important;
        border: 1.5px solid var(--primary-2) !important;
        background: var(--surface) !important;
        box-shadow: 0 4px 16px rgba(224, 122, 95, 0.10) !important;
      }
      [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important;
        opacity: 1 !important;
      }
      [data-testid="stChatInput"] textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(242, 164, 136, 0.30) !important;
      }
      /* Send icon inside chat input */
      [data-testid="stChatInputSubmitButton"] { color: var(--primary) !important; }

      /* Buttons — keep text readable in every state (incl. inner span/p) */
      .stButton button, .stDownloadButton button {
        background-color: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 14px; padding: 0.55rem 1.1rem; font-weight: 600;
        transition: all 0.15s ease;
      }
      .stButton button p, .stButton button span, .stButton button div,
      .stDownloadButton button p, .stDownloadButton button span {
        color: #ffffff !important;
      }
      .stButton button:hover, .stDownloadButton button:hover,
      .stButton button:focus, .stButton button:active,
      .stButton button:focus:not(:active) {
        background-color: #C55F45 !important;
        border-color: #C55F45 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(224, 122, 95, 0.30) !important;
      }
      .stButton button:hover p, .stButton button:hover span, .stButton button:hover div,
      .stButton button:focus p, .stButton button:focus span,
      .stButton button:active p, .stButton button:active span {
        color: #ffffff !important;
      }

      /* Sidebar */
      [data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
      [data-testid="stSidebar"] h2 { font-size: 1.05rem; font-weight: 700; }

      /* Source chips */
      .source-card {
        background: #FCF3EE; border: 1px solid var(--border);
        border-radius: 12px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;
        font-size: 0.85rem; color: var(--text);
      }
      .source-card .src-name { font-weight: 600; color: var(--primary); }
      .source-card .src-preview { color: var(--muted); margin-top: 0.2rem; }

      /* File uploader */
      [data-testid="stFileUploaderDropzone"] {
        background: #FCF3EE; border: 1.5px dashed var(--primary-2);
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

# --- Sidebar: document upload + controls ---
with st.sidebar:
    st.markdown("## 📎 Документи")
    st.caption("Качи файл, за да разшириш знанието на асистента.")

    uploaded = st.file_uploader(
        "PDF, TXT, MD или DOCX",
        type=["pdf", "txt", "md", "docx"],
        label_visibility="collapsed",
    )
    if uploaded is not None and st.button("Индексирай документа", use_container_width=True):
        with st.spinner("Обработвам документа…"):
            try:
                resp = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                st.toast(f"{data['file']} е индексиран успешно", icon="✅")
                st.success(
                    f"**Готово! Документът е запазен и индексиран.**\n\n"
                    f"- Файл: `{data['file']}`\n"
                    f"- Части (chunks): **{data['chunks']}**\n"
                    f"- Тип разбиване: `{data['doc_type']}`\n\n"
                    f"Вече можеш да задаваш въпроси за него."
                )
            except requests.RequestException as exc:
                st.error(f"Грешка при качване: {exc}")

    st.divider()
    if st.button("Нов разговор", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# --- Header ---
st.markdown(
    """
    <div class="hero">
      <h1>Твоят документен асистент</h1>
      <p>Задай въпрос — отговарям само на база твоите документи.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📄"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Източници"):
                for s in msg["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-card">
                          <span class="src-name">[{s['index']}] {s['source']}</span>
                          <div class="src-preview">{s['preview']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# --- Chat input ---
if prompt := st.chat_input("Напиши въпроса си…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Мисля…"):
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
                answer = f"Възникна грешка при връзката с бекенда: {exc}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Източници"):
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

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
