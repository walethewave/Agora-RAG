import streamlit as st
import requests

API_URL = "http://localhost:8000"

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
        if not target_index.strip():
            st.warning("Please specify an Index (Namespace).")
        elif not uploaded_file:
            st.warning("Please upload a TXT or PDF file.")
        else:
            with st.spinner(f"Uploading '{uploaded_file.name}' to index '{target_index.strip()}'..."):
                try:
                    mime = "text/plain" if uploaded_file.name.endswith(".txt") else "application/pdf"
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime)}
                    form_data = {"entity_id": target_index.strip()}
                    resp = requests.post(
                        f"{API_URL}/insert-doc-vector-db",
                        files=files,
                        data=form_data,
                        timeout=120,
                    )
                    data = resp.json()
                    if data.get("responseCode") == "00":
                        st.success(f"✓ Upload started! Task ID: `{data.get('data', {}).get('task_id', 'unknown')}`")
                        st.info("Processing in background — check Task Status tab or poll /task-status.")
                    else:
                        st.error(f"Error: {data.get('responseMessage', 'Unknown Upload Error')}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

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
        with st.spinner("Fetching statistics..."):
            try:
                resp = requests.get(
                    f"{API_URL}/stats",
                    params={"entity_id": stats_index.strip()} if stats_index.strip() else {},
                    timeout=30,
                )
                data = resp.json()
                if data.get("responseCode") == "00":
                    stats_data = data.get("data", {})
                    
                    if stats_index.strip():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Namespace", stats_data.get("namespace", "Unknown"))
                        with col2:
                            st.metric("Vector Count", stats_data.get("total_vectors", 0))
                        
                        st.info(f"Index: {stats_data.get('index_name', 'Unknown')}")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Vectors", stats_data.get("total_vectors", 0))
                        with col2:
                            st.metric("Index Fullness", f"{stats_data.get('index_fullness', 0):.1%}")
                        with col3:
                            st.metric("Embedding Dimension", stats_data.get("embedding_dimension", 1536))
                        
                        # Show all namespaces
                        namespaces = stats_data.get("namespaces", {})
                        if namespaces:
                            st.subheader("Namespaces")
                            for ns_name, ns_info in namespaces.items():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**{ns_name}**")
                                with col2:
                                    st.metric("Vectors", ns_info.get("vector_count", 0))
                else:
                    st.error(f"Error: {data.get('responseMessage', 'Unknown Error')}")
            except Exception as e:
                st.error(f"Connection error: {e}")

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
        if not clear_index.strip():
            st.warning("Please specify the Index to clear.")
        elif confirm_text.strip().upper() != "YES":
            st.warning("You must type 'YES' to confirm deletion.")
        else:
            with st.spinner(f"Clearing index '{clear_index.strip()}'..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/reset-vector-db",
                        data={"entity_id": clear_index.strip(), "confirm": "YES"},
                        timeout=120,
                    )
                    data = resp.json()
                    if data.get("responseCode") == "00":
                        st.success(f"✓ Successfully cleared all data from index '{clear_index.strip()}'.")
                    else:
                        st.error(f"Error: {data.get('responseMessage', 'Unknown Reset Error')}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
