import os
import time
import datetime

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from openai import OpenAI
from rag import HybridRAGPipeline
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AP Elections Intelligence",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Royal Purple CSS ──────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }

/* Keep the sidebar collapse/expand toggle always visible */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    background: #171724 !important;
    border: 1px solid #2D2D4A !important;
    border-radius: 0 8px 8px 0 !important;
    color: #A78BFA !important;
}
[data-testid="collapsedControl"]:hover {
    background: #2D1B69 !important;
    border-color: #8B5CF6 !important;
}

.stApp {
    background-color: #0F0F17;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.main .block-container {
    max-width: 860px !important;
    padding: 0 1.5rem 7rem !important;
    margin: 0 auto !important;
}

/* ── Sidebar — always visible, never collapses ── */
[data-testid="stSidebar"] {
    background-color: #0C0C14 !important;
    border-right: 1px solid #1C1C2C !important;
    transform: translateX(0px) !important;
    margin-left: 0 !important;
    min-width: 244px !important;
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
}
/* Override the aria-expanded=false state Streamlit uses to collapse */
section[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: 0 !important;
    transform: translateX(0px) !important;
    min-width: 244px !important;
    display: flex !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 22px 16px !important; }

/* Hide the collapse toggle button so it can't be clicked again */
[data-testid="collapsedControl"] { display: none !important; }
button[kind="header"] { display: none !important; }

.sb-logo { display: flex; align-items: center; gap: 9px; }
.sb-logo-icon {
    width: 32px; height: 32px;
    background: #1E1240; border: 1px solid #2D1B69;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.sb-title { font-size: .95rem; font-weight: 700; color: #F9FAFB; }
.sb-tagline { font-size: 10.5px; color: #9CA3AF; margin: 3px 0 0 41px; line-height: 1.4; }
.sb-div { height: 1px; background: #1C1C2C; margin: 16px 0; }
.sb-sec {
    font-size: 9px; font-weight: 700; letter-spacing: 1.6px;
    text-transform: uppercase; color: #9CA3AF; margin-bottom: 9px;
}
.year-row { display: flex; flex-wrap: wrap; gap: 5px; }
.yr {
    background: #14142A; color: #D1D5DB;
    border: 1px solid #2D2D4A; border-radius: 6px;
    padding: 3px 10px; font-size: 11px; font-weight: 600;
}
.yr.on { background: #2D1B69; color: #EDE9FE; border-color: #8B5CF6; }

/* sidebar / hero buttons */
div[data-testid="stButton"] > button {
    background: #14142A !important;
    color: #D1D5DB !important;
    border: 1px solid #2D2D4A !important;
    border-radius: 8px !important;
    font-size: 11.5px !important;
    padding: 7px 12px !important;
    transition: background .12s, border-color .12s, color .12s !important;
    white-space: normal !important;
    text-align: left !important;
    line-height: 1.4 !important;
    font-weight: 400 !important;
    box-shadow: none !important;
    transform: none !important;
}
div[data-testid="stButton"] > button:hover {
    background: #2D1B69 !important;
    border-color: #8B5CF6 !important;
    color: #EDE9FE !important;
    box-shadow: none !important;
    transform: none !important;
}
.hist-empty { font-size: 11.5px; color: #6B7280; font-style: italic; }

/* ── App header banner (full-width, no card box) ── */
.app-header {
    background: linear-gradient(130deg, #0D0D1E 0%, #12122A 55%, #0A0A1A 100%);
    border-bottom: 1px solid #2D1B69;
    border-radius: 0;
    padding: 22px calc(1.5rem + 12px);
    margin: -2rem -1.5rem 18px -1.5rem;
    display: flex; align-items: center; justify-content: space-between;
    overflow: hidden;
}
.ah-title {
    font-size: 1.3rem; font-weight: 700; color: #F3F4F6;
    margin: 0 0 5px;
    display: flex; align-items: center; gap: 10px;
}
.ah-icon {
    width: 30px; height: 30px;
    background: rgba(139,92,246,.15);
    border: 1px solid rgba(139,92,246,.3);
    border-radius: 8px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 15px;
}
.ah-sub { font-size: 12px; color: #9CA3AF; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #171724 !important;
    border: 1px solid #1E1E2E !important;
    border-radius: 12px !important;
    padding: 6px 10px !important;
    margin: 8px 0 !important;
}
[data-testid="stChatMessage"] p { color: #D1D5DB; line-height: 1.72; }
[data-testid="stChatMessage"] strong { color: #F3F4F6; }
[data-testid="stChatMessage"] li { color: #D1D5DB; }
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 { color: #A78BFA; }
[data-testid="stChatMessage"] code {
    background: #1A1A30; color: #8B5CF6;
    border-radius: 4px; padding: 1px 5px; font-size: 12px;
}

/* ── Chat input — purple focus, no red ── */
[data-testid="stChatInput"] {
    background: #171724 !important;
    border: 1px solid #2D2D4A !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 2px rgba(139,92,246,0.25) !important;
}
[data-testid="stChatInput"] textarea {
    color: #D1D5DB !important;
    caret-color: #8B5CF6 !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #3A3A5C !important; }
/* Send button — always purple, never red */
[data-testid="stChatInputSubmitButton"] {
    background-color: #8B5CF6 !important;
    border-color: #8B5CF6 !important;
}
[data-testid="stChatInputSubmitButton"]:hover {
    background-color: #7C3AED !important;
    border-color: #7C3AED !important;
}
/* Kill any red/error overrides from BaseWeb */
div[data-baseweb="textarea"] { border-color: #2D2D4A !important; }
div[data-baseweb="textarea"]:focus-within { border-color: #8B5CF6 !important; }
*:focus { outline-color: #8B5CF6 !important; }
* { --primary: #8B5CF6 !important; }

/* ── st.status ── */
[data-testid="stStatusWidget"] {
    background: #12121E !important;
    border: 1px solid #2D1B69 !important;
    border-radius: 10px !important;
    color: #8B5CF6 !important;
    font-size: 13px !important;
}

/* ── Source cards ── */
.src-wrap { margin: 14px 0 0; }
.src-hdr { display: flex; align-items: center; gap: 8px; margin-bottom: 9px; }
.src-lbl {
    font-size: 9.5px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #6B7280;
}
.src-cnt {
    background: #2D1B69; color: #A78BFA;
    border-radius: 20px; padding: 1px 8px;
    font-size: 10px; font-weight: 700;
}
.src-scroll {
    display: flex; gap: 8px;
    overflow-x: auto; padding-bottom: 6px;
    scrollbar-width: thin; scrollbar-color: #2D2D4A transparent;
}
.src-scroll::-webkit-scrollbar { height: 3px; }
.src-scroll::-webkit-scrollbar-thumb { background: #2D2D4A; border-radius: 3px; }

/* Source cards — click-to-reveal below row, accordion */
.sc-card {
    flex-shrink: 0; width: 150px; height: 88px;
    background: #1A1A28; border: 1px solid #24243A;
    border-radius: 10px; padding: 10px 11px;
    cursor: pointer; user-select: none; overflow: hidden;
    box-sizing: border-box;
    transition: background .12s, border-color .12s;
}
.sc-card:hover { background: #22223A; border-color: #3D2D80; }
.sc-card.sc-on { background: #2D1B69; border-color: #8B5CF6; }
.sc-card.sc-on .sc-title,
.sc-card.sc-on .sc-meta { color: #EDE9FE; }
.sc-icon { font-size: 17px; display: block; margin-bottom: 5px; }
.sc-title {
    font-size: 11px; font-weight: 600; color: #D1D5DB;
    line-height: 1.3; margin-bottom: 4px;
    display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden;
}
.sc-meta { font-size: 10px; color: #6B7280; }
.sc-body {
    margin-top: 8px; padding: 12px 14px;
    background: #0D0D20; border: 1px solid #8B5CF6;
    border-radius: 10px; font-size: 11.5px; color: #9CA3AF;
    white-space: pre-wrap; max-height: 220px; overflow-y: auto;
    line-height: 1.6;
}

/* ── Message meta / timer ── */
.msg-meta {
    font-size: 10.5px; color: #374151;
    margin: 8px 0 0; display: flex; align-items: center; gap: 8px;
}
.msg-timer { color: #8B5CF6; font-weight: 500; }
.user-ts { font-size: 10px; color: #4B5563; text-align: right; margin-top: 4px; }

/* ── Action buttons ── */
.actions { display: flex; gap: 5px; margin: 6px 0 0; }
.act-btn {
    background: none; border: 1px solid #1E1E2E;
    border-radius: 6px; padding: 3px 7px;
    font-size: 12px; color: #4B5563; cursor: pointer;
    transition: border-color .12s, color .12s; line-height: 1;
}
.act-btn:hover { border-color: #8B5CF6; color: #A78BFA; }
.act-btn.voted-up { border-color: #8B5CF6; color: #A78BFA; background: rgba(139,92,246,0.15); }
.act-btn.voted-down { border-color: #4B5563; color: #6B7280; background: rgba(75,85,99,0.15); }

/* Selected source card button (primary type) */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #2D1B69 !important;
    border-color: #8B5CF6 !important;
    color: #EDE9FE !important;
    min-height: 80px !important;
}
div[data-testid="stButton"] > button[kind="secondaryFormSubmit"],
div[data-testid="stButton"] > button[kind="secondary"] {
    min-height: 80px !important;
}

/* ── Follow-up chips ── */
.fu-label { font-size: 11px; color: #4B5563; margin: 14px 0 8px; font-weight: 500; }

/* ── Hero / empty state ── */
.hero { text-align: center; padding: 50px 20px 32px; }
.hero-ic { font-size: 2.8rem; display: block; margin-bottom: 14px; }
.hero-h {
    font-size: 1.7rem; font-weight: 700; color: #F3F4F6;
    margin: 0 0 10px; letter-spacing: -.5px;
}
.hero-p { font-size: .85rem; color: #6B7280; max-width: 400px; margin: 0 auto 22px; line-height: 1.65; }
.hero-pills { display: flex; justify-content: center; flex-wrap: wrap; gap: 7px; margin-bottom: 32px; }
.hero-pill {
    background: #14142A; color: #8B5CF6;
    border: 1px solid #2D1B69; border-radius: 20px; padding: 3px 13px; font-size: 11px;
}
.su-lbl {
    font-size: 10px; font-weight: 700; letter-spacing: 1.4px;
    text-transform: uppercase; color: #3A3A5C; margin-bottom: 12px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2D2D4A; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #8B5CF6; }
</style>
""", unsafe_allow_html=True)

# ── City silhouette SVG (Royal Purple) ───────────────────────
CITY_SVG = """
<svg width="165" height="70" viewBox="0 0 165 70" fill="#8B5CF6" xmlns="http://www.w3.org/2000/svg">
  <!-- Central temple spire -->
  <polygon points="82,2 78,14 86,14" opacity=".72"/>
  <rect x="78" y="14" width="8" height="4" opacity=".65"/>
  <polygon points="80,18 71,28 93,28" opacity=".55"/>
  <rect x="71" y="28" width="22" height="4" opacity=".5"/>
  <rect x="67" y="32" width="30" height="38" opacity=".2"/>
  <rect x="72" y="48" width="5" height="22" opacity=".32"/>
  <rect x="88" y="48" width="5" height="22" opacity=".32"/>
  <rect x="77" y="41" width="10" height="7" opacity=".38"/>
  <rect x="78" y="55" width="8" height="15" fill="#0F0F17" opacity=".75"/>
  <!-- Left minaret -->
  <polygon points="35,10 31,22 39,22" opacity=".5"/>
  <rect x="31" y="22" width="8" height="48" opacity=".24"/>
  <rect x="29" y="35" width="12" height="3" opacity=".3"/>
  <!-- Right minaret -->
  <polygon points="130,10 126,22 134,22" opacity=".5"/>
  <rect x="126" y="22" width="8" height="48" opacity=".24"/>
  <rect x="124" y="35" width="12" height="3" opacity=".3"/>
  <!-- Left buildings -->
  <rect x="5" y="43" width="18" height="27" opacity=".17"/>
  <rect x="9" y="37" width="8" height="6" opacity=".22"/>
  <rect x="45" y="47" width="16" height="23" opacity=".15"/>
  <rect x="49" y="41" width="6" height="6" opacity=".2"/>
  <!-- Right buildings -->
  <rect x="106" y="47" width="16" height="23" opacity=".15"/>
  <rect x="110" y="41" width="6" height="6" opacity=".2"/>
  <rect x="145" y="41" width="16" height="29" opacity=".17"/>
  <rect x="150" y="35" width="7" height="6" opacity=".22"/>
</svg>
"""


# ── Init ─────────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    return HybridRAGPipeline()


@st.cache_data(show_spinner=False)
def get_followups(question: str, answer_snippet: str) -> list:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=110,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate exactly 3 short follow-up questions (max 8 words each) "
                        "about Andhra Pradesh elections based on this Q&A. "
                        "Return one question per line, no bullets or numbers."
                    ),
                },
                {"role": "user", "content": f"Q: {question}\nA: {answer_snippet}"},
            ],
        )
        lines = resp.choices[0].message.content.strip().split("\n")
        return [l.strip().lstrip("•-0123456789. ").strip() for l in lines if l.strip()][:3]
    except Exception:
        return []


rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ── Helpers ──────────────────────────────────────────────────
def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def source_title(s: dict) -> str:
    year = s.get("year", "?")
    stype = s.get("type", "")
    if stype == "election_results":
        return f"{year} Election Results"
    if stype == "analysis":
        return f"{year} Election Analysis"
    src = s.get("source", "")
    return os.path.basename(src) if src else f"Source {year}"


def render_sources(sources: list, uid: str = "0"):
    if not sources:
        return

    state_key = f"src_sel_{uid}"
    if state_key not in st.session_state:
        st.session_state[state_key] = None

    icons = {"analysis": "📄", "election_results": "📊"}

    # Header
    st.markdown(
        f'<div class="src-hdr">'
        f'<span class="src-lbl">Sources</span>'
        f'<span class="src-cnt">{len(sources)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # One column per source — buttons look like cards via CSS
    cols = st.columns(len(sources))
    for i, (col, s) in enumerate(zip(cols, sources)):
        icon = icons.get(s.get("type", ""), "📁")
        title = source_title(s)
        year = s.get("year", "?")
        chunk = s.get("chunk_number", "?")
        is_sel = st.session_state[state_key] == i
        label = f"{icon}\n\n**{title}**\n\nYear: {year} · Chunk: {chunk}"
        with col:
            if st.button(
                label,
                key=f"src_{uid}_{i}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                st.session_state[state_key] = None if is_sel else i
                st.rerun()

    # Chunk text shown below cards when one is selected
    sel = st.session_state[state_key]
    if sel is not None and sel < len(sources):
        chunk_text = sources[sel].get("chunk_text", "")
        st.markdown(
            f'<div class="sc-body">{esc(chunk_text)}</div>',
            unsafe_allow_html=True,
        )


def render_msg_meta(elapsed: float = None):
    if elapsed is None:
        return
    st.markdown(
        f'<div class="msg-meta">'
        f'<span class="msg-timer">✓ Completed in {elapsed:.1f}s</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_actions():
    st.markdown(
        '<div class="actions">'
        '<button class="act-btn" title="Copy" '
        'onclick="var t=this.closest(\'[data-testid=stChatMessage]\');"'
        '>📋</button>'
        '<button class="act-btn" title="Helpful" '
        'onclick="var b=this,p=b.parentNode,o=p.querySelector(\'.voted-down\');'
        'if(o)o.classList.remove(\'voted-down\');'
        'b.classList.toggle(\'voted-up\');"'
        '>👍</button>'
        '<button class="act-btn" title="Not helpful" '
        'onclick="var b=this,p=b.parentNode,o=p.querySelector(\'.voted-up\');'
        'if(o)o.classList.remove(\'voted-up\');'
        'b.classList.toggle(\'voted-down\');"'
        '>👎</button>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_followups(followups: list, key_prefix: str = ""):
    if not followups:
        return
    st.markdown('<div class="fu-label">Continue exploring →</div>', unsafe_allow_html=True)
    cols = st.columns(len(followups))
    for col, q in zip(cols, followups):
        if col.button(q, key=f"{key_prefix}_{hash(q)}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sb-logo">'
        '<div class="sb-logo-icon">🗳️</div>'
        '<div class="sb-title">AP Elections RAG</div>'
        "</div>"
        '<div class="sb-tagline">AI-powered election intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sb-div'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-sec'>Elections Covered</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='year-row'>"
        "<span class='yr'>2004</span>"
        "<span class='yr'>2008</span>"
        "<span class='yr'>2014</span>"
        "<span class='yr'>2019</span>"
        "<span class='yr on'>2024</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sb-div'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-sec'>Recent Conversations</div>", unsafe_allow_html=True)

    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    if user_msgs:
        for msg in reversed(user_msgs[-10:]):
            q = msg["content"]
            if st.button(q, key=f"hist_{hash(q)}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()
    else:
        st.markdown(
            "<div class='hist-empty'>No conversations yet.</div>",
            unsafe_allow_html=True,
        )


# ── App header banner ─────────────────────────────────────────
st.markdown(
    f'<div class="app-header">'
    f"<div>"
    f'<div class="ah-title">'
    f'<span class="ah-icon">🗳️</span>AP Elections Intelligence'
    f"</div>"
    f'<div class="ah-sub">Ask anything about Andhra Pradesh elections (2004–2024)</div>'
    f"</div>"
    f"{CITY_SVG}"
    f"</div>",
    unsafe_allow_html=True,
)


# ── Resolve query ─────────────────────────────────────────────
chat_input = st.chat_input("Ask a question about AP elections…")
user_query = chat_input or st.session_state.pending_query
if st.session_state.pending_query and user_query == st.session_state.pending_query:
    st.session_state.pending_query = None


# ── Empty state ───────────────────────────────────────────────
if not st.session_state.messages and not user_query:
    st.markdown(
        '<div class="hero">'
        '<span class="hero-ic">🗳️</span>'
        '<div class="hero-h">AP Elections RAG</div>'
        '<div class="hero-p">'
        "AI-powered Andhra Pradesh election intelligence across five election cycles — 2004 to 2024."
        "</div>"
        '<div class="hero-pills">'
        "<span class='hero-pill'>2004 – 2024</span>"
        "<span class='hero-pill'>Hybrid RAG</span>"
        "<span class='hero-pill'>GPT-4o-mini</span>"
        "<span class='hero-pill'>Hallucination-resistant</span>"
        "</div>"
        '<div class="su-lbl">Try one of these</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    chips = [
        "Who won the 2024 AP elections?",
        "Compare TDP performance across years",
        "What happened in the 2014 elections?",
        "Which party had the highest vote share?",
    ]
    c1, c2 = st.columns(2)
    pairs = [(c1, chips[0]), (c2, chips[1]), (c1, chips[2]), (c2, chips[3])]
    for col, q in pairs:
        if col.button(q, key=f"hero_{hash(q)}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()


# ── Render conversation history ───────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    is_last = i == len(st.session_state.messages) - 1

    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**{msg['content']}**")
            if msg.get("ts"):
                st.markdown(
                    f'<div class="user-ts">{msg["ts"]}</div>',
                    unsafe_allow_html=True,
                )
    else:
        with st.chat_message("assistant", avatar="🔮"):
            st.markdown(msg["content"])
            render_msg_meta(msg.get("elapsed"))
            render_actions()
            render_sources(msg.get("sources", []), uid=f"h{i}")
            if is_last:
                render_followups(msg.get("followups", []), key_prefix=f"hist_{i}")


# ── Process new query ─────────────────────────────────────────
if user_query:
    ts = datetime.datetime.now().strftime("%I:%M %p")

    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**{user_query}**")
        st.markdown(f'<div class="user-ts">{ts}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🔮"):
        with st.status("Working on it…", expanded=True) as status:
            st.write("🔍 Understanding your question…")
            st.write("⚡ Running hybrid retrieval (semantic + BM25)…")
            t0 = time.time()
            answer, sources = rag.generate_answer(user_query)
            elapsed = time.time() - t0
            followups = get_followups(user_query, answer[:400])
            st.write("✨ Grounded response ready")
            status.update(
                label=f"Done · {elapsed:.1f}s",
                state="complete",
                expanded=False,
            )

        st.markdown(answer)
        render_msg_meta(elapsed)
        render_actions()
        render_sources(sources, uid=f"n{len(st.session_state.messages)}")
        render_followups(followups, key_prefix=f"new_{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {"role": "user", "content": user_query, "ts": ts}
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "followups": followups,
            "elapsed": elapsed,
        }
    )
