import json
import streamlit as st
import requests
import redis
import uuid
from datetime import datetime
from src.utils import load_project_env, read_env_value
from src.simplified_rag import SimplifiedRAG

# ── RAG System Initialization ──────────────────────────────────────────────────
@st.cache_resource
def get_rag_system():
    import os
    # Inject Streamlit secrets into env vars for SimplifiedRAG (cloud deployment)
    for key in ["GEMINI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "REDIS_URL"]:
        try:
            os.environ.setdefault(key, st.secrets[key])
        except (KeyError, FileNotFoundError):
            pass
    try:
        return SimplifiedRAG()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

rag_system = get_rag_system()
ENTITY_ID = st.query_params.get("entity_id", "policy")

st.set_page_config(
    page_title="Agora",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Upstash Redis ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_redis():
    # Try Streamlit secrets first (for cloud deployment), then .env file
    url = None
    try:
        url = st.secrets["REDIS_URL"]
    except (KeyError, FileNotFoundError):
        env_file = load_project_env()
        url = read_env_value("REDIS_URL", env_file)
    if not url:
        return None
    try:
        client = redis.from_url(url, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None

rc = get_redis()
SESSIONS_INDEX = "agora:sessions"

def upstash_save_message(session_id: str, role: str, content: str, sources: list = None):
    if not rc:
        return
    msg = json.dumps({"role": role, "content": content, "sources": sources or [], "ts": datetime.now().isoformat()})
    key = f"agora:{session_id}:messages"
    rc.rpush(key, msg)
    rc.expire(key, 86400 * 7)

def upstash_load_messages(session_id: str) -> list:
    if not rc:
        return []
    return [json.loads(m) for m in rc.lrange(f"agora:{session_id}:messages", 0, -1)]

def upstash_save_session(session_id: str, title: str):
    if not rc:
        return
    rc.hset(SESSIONS_INDEX, session_id, json.dumps({"session_id": session_id, "title": title, "ts": datetime.now().isoformat()}))
    rc.expire(SESSIONS_INDEX, 86400 * 7)

def upstash_update_title(session_id: str, title: str):
    if not rc:
        return
    existing = rc.hget(SESSIONS_INDEX, session_id)
    if existing:
        meta = json.loads(existing)
        meta["title"] = title
        rc.hset(SESSIONS_INDEX, session_id, json.dumps(meta))

def upstash_load_sessions() -> list:
    if not rc:
        return []
    sessions = [json.loads(v) for v in rc.hgetall(SESSIONS_INDEX).values()]
    # Only show sessions that have messages
    active = []
    for s in sessions:
        msgs = rc.lrange(f"agora:{s['session_id']}:messages", 0, 0)
        if msgs:
            active.append(s)
    return sorted(active, key=lambda x: x.get("ts", ""), reverse=True)

def upstash_delete_session(session_id: str):
    if not rc:
        return
    rc.hdel(SESSIONS_INDEX, session_id)
    rc.delete(f"agora:{session_id}:messages")

# ── API helpers (Now Local Calls) ───────────────────────────────────────────────
def api_create_session() -> str:
    return str(uuid.uuid4())

def api_ask(question: str, session_id: str) -> tuple[str, list]:
    if not rag_system:
        return "❌ RAG system not initialized. Check your environment variables.", []
    
    try:
        result = rag_system.ask_questions(
            question=question,
            session_id=session_id,
            namespace=ENTITY_ID,
        )
        if result.get("success"):
            answer = result.get("answer", "No answer returned.")
            sources = result.get("sources", [])
            return answer, sources
        return f"⚠️ {result.get('error', 'Something went wrong')}", []
    except Exception as e:
        return f"❌ Error: {str(e)}", []

# ── Session state init ─────────────────────────────────────────────────────────
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "local_messages" not in st.session_state:
    st.session_state.local_messages = []

# Auto-create a session on first load so chat input is immediately available
if st.session_state.active_session_id is None:
    sid = api_create_session()
    if sid:
        st.session_state.active_session_id = sid
        st.session_state.local_messages = []
        # Don't save to Upstash yet — only save when first message is sent

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Agora")

    if st.button("＋ New Chat", type="primary", use_container_width=True):
        # Only create new session if current one already has messages
        current_has_messages = len(st.session_state.local_messages) > 0
        if current_has_messages:
            sid = api_create_session()
            if sid:
                st.session_state.active_session_id = sid
                st.session_state.local_messages = []
                st.rerun()
            else:
                st.error("Could not reach API")
        # If current session is empty, just clear and reuse it (no-op)

    if st.button("⚙️ Admin Panel", type="secondary", use_container_width=True):
        st.switch_page("pages/admin.py")

    st.divider()
    st.markdown("**History**")

    sessions = upstash_load_sessions()
    if not sessions:
        st.caption("No conversations yet.")
    else:
        for s in sessions:
            sid = s["session_id"]
            title = s.get("title", "Untitled")
            ts = s.get("ts", "")[:10]
            is_active = sid == st.session_state.active_session_id

            col1, col2 = st.columns([5, 1])
            with col1:
                label = f"▶ {title[:26]}" if is_active else title[:30]
                if st.button(label, key=f"s_{sid}", use_container_width=True,
                             type="primary" if is_active else "secondary", help=ts):
                    st.session_state.active_session_id = sid
                    st.session_state.local_messages = upstash_load_messages(sid)
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"d_{sid}", help="Delete"):
                    upstash_delete_session(sid)
                    if st.session_state.active_session_id == sid:
                        # Create a fresh session after deletion
                        new_sid = api_create_session()
                        st.session_state.active_session_id = new_sid
                        st.session_state.local_messages = []
                    st.rerun()

# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("Agora")

if not st.session_state.active_session_id:
    sid = api_create_session()
    st.session_state.active_session_id = sid
    st.session_state.local_messages = []

# Show welcome only when no messages
if not st.session_state.local_messages:
    st.markdown("## 👋 How can I help you today?")
    st.caption(f"Ask me anything about the AI governance documents · namespace: `{ENTITY_ID}`")

# Display messages
for msg in st.session_state.local_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    st.caption(f"**{src.get('filename','')}** · score: {src.get('relevance_score',0):.3f}")

# Chat input
if prompt := st.chat_input("Message Agora..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    user_msg = {"role": "user", "content": prompt, "sources": [], "ts": datetime.now().isoformat()}
    st.session_state.local_messages.append(user_msg)

    # Save session to Upstash on first message
    is_first_message = sum(1 for m in st.session_state.local_messages if m["role"] == "user") == 1
    if is_first_message:
        title = prompt[:40] + ("..." if len(prompt) > 40 else "")
        upstash_save_session(st.session_state.active_session_id, title)
    upstash_save_message(st.session_state.active_session_id, "user", prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = api_ask(prompt, st.session_state.active_session_id)
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources", expanded=False):
                for src in sources:
                    st.caption(f"**{src.get('filename','')}** · score: {src.get('relevance_score',0):.3f}")

    asst_msg = {"role": "assistant", "content": answer, "sources": sources, "ts": datetime.now().isoformat()}
    st.session_state.local_messages.append(asst_msg)
    upstash_save_message(st.session_state.active_session_id, "assistant", answer, sources)
    st.rerun()
