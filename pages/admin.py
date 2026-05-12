import streamlit as st
import requests
from src.simplified_rag import SimplifiedRAG

# ── RAG System Initialization ──────────────────────────────────────────────────
@st.cache_resource
def get_rag_system():
    try:
        return SimplifiedRAG()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

rag_system = get_rag_system()

st.set_page_config(
    page_title="Agora Admin",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Agora")
    if st.button("▶ Back to Chat", type="primary", use_container_width=True):
        st.switch_page("chat.py")
    st.divider()
    st.markdown("**History**")
    st.caption("Switch to Chat to see session history.")

st.title("⚙️ Agora Admin")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_upload, tab_stats, tab_clear = st.tabs(["📤 Upload Document", "📊 Database Stats", "🗑️ Clear Database"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload Document
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### Upload Document to Vector Database")
    st.caption("Upload a TXT file. Specify the index (namespace) to group multiple documents together.")

    with st.form("upload_document_form", clear_on_submit=False):
        target_index = st.text_input("Index (Namespace)", placeholder="e.g. policy", help="Documents uploaded with the same index will belong to the same grouping.")
        uploaded_file = st.file_uploader("Choose a TXT file", type=["txt"], key="single_file")

        submitted = st.form_submit_button("Upload Document", type="primary", use_container_width=True)

    if submitted:
        if not rag_system:
            st.error("RAG system not initialized.")
        elif not target_index.strip():
            st.warning("Please specify an Index (Namespace).")
        elif not uploaded_file:
            st.warning("Please upload a file.")
        else:
            with st.spinner(f"Processing '{uploaded_file.name}' into index '{target_index.strip()}'..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    result = rag_system.add_to_existing_collection(
                        file_bytes=file_bytes,
                        filename=uploaded_file.name,
                        namespace=target_index.strip(),
                    )
                    if result.get('success'):
                        st.success(f"✓ Successfully indexed '{uploaded_file.name}'!")
                        st.json(result)
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Processing error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Database Stats
# ═══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown("### Database Statistics")
    st.caption("View vector count and other metrics for a specific index.")

    with st.form("stats_form"):
        stats_index = st.text_input("Index (Namespace)", placeholder="e.g. policy", help="Leave empty to see global stats")
        stats_btn = st.form_submit_button("Get Stats", type="primary", use_container_width=True)

    if stats_btn:
        if not rag_system:
            st.error("RAG system not initialized.")
        else:
            with st.spinner("Fetching statistics..."):
                try:
                    # stats = rag_system.get_database_stats(namespace=stats_index.strip() if stats_index.strip() else None)
                    # Use the index.describe_index_stats() directly to match app.py logic
                    raw_stats = rag_system.index.describe_index_stats()
                    
                    if stats_index.strip():
                        ns_stats = raw_stats.get("namespaces", {}).get(stats_index.strip(), {})
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Namespace", stats_index.strip())
                        with col2:
                            st.metric("Vector Count", ns_stats.get("vector_count", 0))
                        
                        st.info(f"Index: {rag_system.index_name}")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Vectors", raw_stats.get("total_vector_count", 0))
                        with col2:
                            st.metric("Index Fullness", f"{raw_stats.get('index_fullness', 0):.1%}")
                        with col3:
                            st.metric("Embedding Dimension", raw_stats.get("dimension", 1536))
                        
                        # Show all namespaces
                        namespaces = raw_stats.get("namespaces", {})
                        if namespaces:
                            st.subheader("Namespaces")
                            for ns_name, ns_info in namespaces.items():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**{ns_name}**")
                                with col2:
                                    st.metric("Vectors", ns_info.get("vector_count", 0))
                except Exception as e:
                    st.error(f"Error fetching stats: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Clear Database
# ═══════════════════════════════════════════════════════════════════════════════
with tab_clear:
    st.markdown("### Clear Database")
    st.caption("Wipe all data from a specific index (namespace).")

    with st.form("clear_db_form"):
        clear_index = st.text_input("Index to Clear (Namespace)", placeholder="e.g. policy")
        confirm_text = st.text_input("Type 'YES' to confirm", placeholder="YES")
        
        clear_btn = st.form_submit_button("🚨 Delete All Data in Index", type="primary", use_container_width=True)

    if clear_btn:
        if not rag_system:
            st.error("RAG system not initialized.")
        elif not clear_index.strip():
            st.warning("Please specify the Index to clear.")
        elif confirm_text.strip().upper() != "YES":
            st.warning("You must type 'YES' to confirm deletion.")
        else:
            with st.spinner(f"Clearing index '{clear_index.strip()}'..."):
                try:
                    result = rag_system.reset_vector_database(namespace=clear_index.strip())
                    if result.get('success'):
                        st.success(f"✓ Successfully cleared all data from index '{clear_index.strip()}'.")
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error clearing database: {e}")
