import json
import streamlit as st
import requests
from datetime import datetime

LAMBDA_URL = "https://gij3liro3kouweizzludyyhbe40dsxcw.lambda-url.us-east-1.on.aws"
LOCAL_URL = "http://localhost:8000"

# Toggle: set to LOCAL_URL for local dev, LAMBDA_URL for production
API_URL = LAMBDA_URL

# Streaming only works with local uvicorn (Mangum/Lambda doesn't support SSE)
USE_STREAMING = API_URL == LOCAL_URL

st.set_page_config(
    page_title="Qorpy",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start collapsed for cleaner look
)

# Modern, clean CSS with proper alignment and smooth interactions
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hide all Streamlit chrome */
    #MainMenu, header, footer, .stDeployButton {
        visibility: hidden;
        display: none !important;
    }
    
    /* Main background - clean white/gray modern look */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #fafafa 0%, #f5f5f5 100%);
    }
    
    [data-testid="stMain"] {
        background: transparent;
        max-width: 900px;
        margin: 0 auto;
    }
    
    .block-container {
        padding: 0 1rem 2rem 1rem !important;
        max-width: 800px !important;
        padding-top: 72px !important;
    }
    
    /* Sidebar - sleek dark mode */
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #222 !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }
    
    [data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem !important;
    }
    
    /* New Chat Button - prominent */
    .stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #f0f0f0 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* Chat Messages - PROPER ALIGNMENT */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
        gap: 0.75rem !important;
    }
    
    /* USER messages - RIGHT aligned */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background: #111111 !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 0.875rem 1.25rem !important;
        max-width: 80% !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        border: none !important;
    }
    /* strip inner wrapper — prevents double bubble */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        width: 100% !important;
    }
    /* white text for user messages */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] p,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] * {
        color: #ffffff !important;
    }

    /* ASSISTANT messages - LEFT aligned */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        flex-direction: row !important;
        justify-content: flex-start !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        background: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 0.875rem 1.25rem !important;
        max-width: 80% !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
    }
    /* strip inner wrapper — prevents double bubble */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        width: 100% !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] p,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] * {
        color: #111111 !important;
    }

    /* Message text sizing */
    [data-testid="stChatMessage"] p {
        font-size: 15px !important;
        line-height: 1.6 !important;
        margin: 0 !important;
    }
    
    /* Avatars - modern and clean */
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
        font-size: 14px !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
    }
    
    [data-testid="stChatMessageAvatarAssistant"] {
        background: #f0f0f0 !important;
        color: #666 !important;
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
        font-size: 14px !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* Chat Input - floating modern style */
    [data-testid="stBottom"] {
        background: linear-gradient(to top, #fafafa 60%, transparent) !important;
        padding: 1rem 1rem 2rem 1rem !important;
    }
    
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 24px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
        padding: 0.25rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 4px 24px rgba(102, 126, 234, 0.15) !important;
        transform: translateY(-2px);
    }
    
    [data-testid="stChatInput"] textarea {
        font-size: 15px !important;
        padding: 0.75rem 1rem !important;
        color: #ffffff !important;
    }
    
    [data-testid="stChatInput"] button {
        background: #111 !important;
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        margin: 4px !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stChatInput"] button:hover {
        background: #333 !important;
        transform: scale(1.05);
    }
    
    /* Welcome Screen - centered and beautiful */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 70vh;
        text-align: center;
        padding: 2rem;
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .welcome-logo {
        width: 64px;
        height: 64px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .welcome-title {
        font-size: 32px;
        font-weight: 600;
        color: #111;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .welcome-subtitle {
        font-size: 16px;
        color: #666;
        margin-bottom: 2.5rem;
        max-width: 400px;
        line-height: 1.5;
    }
    
    /* Suggestion chips - clickable and modern */
    .suggestions {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
        max-width: 600px;
    }
    
    .suggestion-chip {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 20px;
        padding: 10px 18px;
        font-size: 14px;
        color: #444;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .suggestion-chip:hover {
        background: #f8f8f8;
        border-color: #667eea;
        color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    /* Sidebar history items */
    .history-item {
        padding: 10px 12px;
        border-radius: 8px;
        font-size: 13px;
        color: #888;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        border: 1px solid transparent;
    }
    
    .history-item:hover {
        background: #1a1a1a;
        color: #fff;
        border-color: #333;
    }
    
    .history-item.active {
        background: #1a1a1a;
        color: #fff;
        border-color: #444;
    }
    
    .sidebar-header {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #666;
        margin: 1.5rem 0 0.75rem 0;
        padding-left: 4px;
    }
    
    /* Spinner - elegant */
    .stSpinner > div {
        border-color: rgba(102, 126, 234, 0.2) !important;
        border-top-color: #667eea !important;
    }
    
    /* Scrollbar - minimal */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #ddd;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #ccc;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .block-container {
            padding: 0 0.75rem 1rem 0.75rem !important;
        }
        
        div[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
            max-width: 90% !important;
        }
        
        .welcome-title {
            font-size: 24px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ── API Function ───────────────────────────────────────────────────────────────

def ask_question(question: str) -> str:
    """Non-streaming fallback — used for suggestion chips."""
    try:
        response = requests.post(
            f"{API_URL}/ask-question",
            json={"question": question},
            timeout=60
        )
        data = response.json()
        if data.get("responseCode") == "00":
            result = data.get("data", {})
            return result.get("answer") or result.get("response") or str(result)
        return f"⚠️ {data.get('responseMessage', 'Something went wrong')}"
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. Please try again."
    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def ask_question_stream(question: str):
    """Generator that yields answer text chunks from the SSE streaming endpoint."""
    try:
        with requests.post(
            f"{API_URL}/ask-question-stream",
            json={"question": question},
            stream=True,
            timeout=60
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    text = chunk.get("text", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue
    except requests.exceptions.Timeout:
        yield "\n\n⏱️ Request timed out. Please try again."
    except Exception as e:
        yield f"\n\n❌ Connection error: {str(e)}"

# ── Session State Management ──────────────────────────────────────────────────

if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    
if "active_id" not in st.session_state:
    st.session_state.active_id = None

def create_new_chat():
    cid = datetime.now().strftime("%Y%m%d%H%M%S%f")
    st.session_state.conversations[cid] = {
        "title": "New conversation", 
        "messages": [],
        "timestamp": datetime.now()
    }
    st.session_state.active_id = cid
    return cid

# Initialize first conversation if none exists
if not st.session_state.conversations:
    create_new_chat()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo area
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid #222;">
        <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 16px;">✨</div>
        <span style="font-size: 18px; font-weight: 600; color: #fff;">Qorpy</span>
    </div>
    """, unsafe_allow_html=True)
    
    # New Chat Button
    if st.button("＋ New Chat", type="secondary", use_container_width=True):
        create_new_chat()
        st.rerun()

    # Admin Page Button
    if st.button("⚙️ Admin Panel", type="secondary", use_container_width=True):
        st.switch_page("pages/admin.py")

    # History Section
    if st.session_state.conversations:
        st.markdown('<div class="sidebar-header">Recent</div>', unsafe_allow_html=True)
        
        # Sort by timestamp, newest first
        sorted_convs = sorted(
            st.session_state.conversations.items(),
            key=lambda x: x[1].get("timestamp", datetime.min),
            reverse=True
        )
        
        for cid, conv in sorted_convs:
            title = conv["title"]
            is_active = cid == st.session_state.active_id
            
            # Use columns for better click handling
            cols = st.columns([1])
            with cols[0]:
                if st.button(
                    f"{'● ' if is_active else '○ '} {title[:25]}{'...' if len(title) > 25 else ''}",
                    key=f"hist_{cid}",
                    help="Click to open conversation",
                    use_container_width=True,
                    type="secondary" if not is_active else "primary"
                ):
                    st.session_state.active_id = cid
                    st.rerun()

# ── Main Chat Interface ────────────────────────────────────────────────────────

st.markdown("""
<div style="
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 9999;
    background: #ffffff;
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 800;
    color: #111111;
    letter-spacing: 0.08em;
    padding: 1rem 2rem;
    border-bottom: 2px solid #111111;
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
    align-items: center;
">
    <span>QORPY FAQ</span>
    <a href="/admin" target="_self" style="
        font-size: 13px;
        font-weight: 600;
        text-transform: none;
        letter-spacing: 0;
        color: #ffffff;
        background: #111111;
        border-radius: 8px;
        padding: 6px 16px;
        text-decoration: none;
    ">⚙️ Admin Panel</a>
</div>
""", unsafe_allow_html=True)

active_conv = st.session_state.conversations[st.session_state.active_id]
messages = active_conv["messages"]

# Welcome State (when no messages)
if not messages:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-logo">✨</div>
        <div class="welcome-title">How can I help you today?</div>
        <div class="welcome-subtitle">Ask me anything about Qorpy — products, pricing, getting started, or troubleshooting.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Clickable suggestion chips using actual buttons
    cols = st.columns(2)
    suggestions = [
        "What is Qorpy?",
        "How do I get started?",
        "What are pricing plans?",
        "How does billing work?",
        "Can I cancel anytime?",
        "Contact support"
    ]
    
    # Create grid of suggestion buttons
    suggestion_cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i % 3]:
            if st.button(suggestion, key=f"sugg_{i}", use_container_width=True, type="secondary"):
                # Simulate user sending this message
                messages.append({"role": "user", "content": suggestion})
                active_conv["title"] = suggestion[:40]
                
                # Get response immediately
                with st.spinner(""):
                    answer = ask_question(suggestion)
                messages.append({"role": "assistant", "content": answer})
                st.rerun()

else:
    # Chat header showing conversation title
    st.markdown(f"""
    <div style="padding: 1rem 0; border-bottom: 1px solid #e0e0e0; margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between;">
        <span style="font-size: 14px; font-weight: 500; color: #666;">{active_conv["title"]}</span>
        <span style="font-size: 12px; color: #999;">{len(messages)//2} messages</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Display existing messages
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ── Chat Input ─────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Message Qorpy...", key="chat_input"):
    # Deduplicate: skip if this exact prompt was already processed this session
    if st.session_state.get("_last_prompt") == prompt:
        st.stop()
    st.session_state["_last_prompt"] = prompt

    # Add user message
    messages.append({"role": "user", "content": prompt})
    
    # Update conversation title on first message
    if len(messages) == 1:
        active_conv["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Stream or block depending on deployment target
    with st.chat_message("assistant"):
        if USE_STREAMING:
            answer = st.write_stream(ask_question_stream(prompt))
        else:
            with st.spinner("Thinking..."):
                answer = ask_question(prompt)
            st.markdown(answer)

    # Save assistant message
    messages.append({"role": "assistant", "content": answer})
    
    # Clear dedup key then rerun to commit messages to history
    st.session_state["_last_prompt"] = None
    st.rerun()