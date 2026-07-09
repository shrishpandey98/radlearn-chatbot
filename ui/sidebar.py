"""
ui/sidebar.py
─────────────
Sidebar component for displaying Knowledge Base status and metrics.
"""
import streamlit as st
import sqlite3
import uuid
from radlearn.config import SQLITE_PATH, EMBEDDING_MODEL
from radlearn.database.conversations import get_all_conversations, get_conversation_messages, delete_conversation
from radlearn.database.citations import get_citations_for_message

def get_db_metrics():
    """Fetch live metrics from SQLite."""
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        
        # Total counts
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT SUM(chunk_count) FROM documents")
        total_chunks = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT SUM(image_count) FROM documents")
        total_images = cursor.fetchone()[0] or 0
        
        # Document statuses
        try:
            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'completed'")
            completed_docs = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'failed'")
            failed_docs = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'pending'")
            pending_docs = cursor.fetchone()[0] or 0
        except sqlite3.OperationalError:
            completed_docs = total_docs
            failed_docs = 0
            pending_docs = 0
            
        conn.close()
        
        return {
            "total_docs": total_docs,
            "total_chunks": total_chunks,
            "total_images": total_images,
            "completed_docs": completed_docs,
            "failed_docs": failed_docs,
            "pending_docs": pending_docs
        }
    except Exception as e:
        return None

def load_conversation(conv_id, session_token):
    messages = get_conversation_messages(conv_id)
    loaded_messages = []
    for msg in messages:
        role = msg["role"]
        m = {
            "id": msg["id"],
            "role": role,
            "content": msg["content"],
            "is_error": False,
            "was_helpful": msg.get("was_helpful"),
        }
        if role == "assistant":
            m["citations"] = get_citations_for_message(msg["id"])
            m["retrieved_chunks"] = [] # Debug chunks won't be loaded
            m["confidence"] = msg["top_similarity_score"]
            m["latency"] = msg["total_time_ms"] / 1000.0 if msg["total_time_ms"] else 0.0
            m["hallucination_risk"] = "Unknown (Historical)"
        loaded_messages.append(m)
    
    st.session_state.session_id = session_token
    st.session_state.conversation_db_id = conv_id
    st.session_state.messages = loaded_messages
    st.rerun()

def render_sidebar():
    with st.sidebar:
        st.title("🏥 RadLearn")
        
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.session_id = str(uuid.uuid4())
            if "conversation_db_id" in st.session_state:
                del st.session_state.conversation_db_id
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("---")
        
        st.subheader("Projects")
        from radlearn.database.projects import get_all_projects, create_project
        projects = get_all_projects()
        
        project_options = {"None": "Global Knowledge Base"}
        for p in projects:
            project_options[p["id"]] = p["name"]
            
        current_project_id = st.session_state.get("project_id", "None")
        if current_project_id not in project_options:
            current_project_id = "None"
            
        selected_project_id = st.selectbox(
            "Select Project",
            options=list(project_options.keys()),
            format_func=lambda x: project_options[x],
            index=list(project_options.keys()).index(current_project_id)
        )
        
        if selected_project_id != current_project_id:
            if selected_project_id == "None":
                if "project_id" in st.session_state:
                    del st.session_state.project_id
            else:
                st.session_state.project_id = selected_project_id
            
            # Start new chat when project changes
            st.session_state.session_id = str(uuid.uuid4())
            if "conversation_db_id" in st.session_state:
                del st.session_state.conversation_db_id
            st.session_state.messages = []
            st.rerun()

        @st.dialog("Create New Project")
        def create_project_dialog():
            with st.form("new_project_form"):
                p_name = st.text_input("Project Name", placeholder="e.g. MSK Board Review")
                p_desc = st.text_area("Description", placeholder="Optional description")
                p_sys = st.text_area("Custom System Instructions", placeholder="e.g. Always respond in bullet points.", height=150)
                submit_project = st.form_submit_button("Create Project")
                if submit_project:
                    if p_name.strip():
                        new_p = create_project({
                            "name": p_name.strip(),
                            "description": p_desc.strip(),
                            "system_instructions": p_sys.strip()
                        })
                        st.session_state.project_id = new_p["id"]
                        st.session_state.session_id = str(uuid.uuid4())
                        if "conversation_db_id" in st.session_state:
                            del st.session_state.conversation_db_id
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error("Project name is required.")

        if st.button("➕ New Project", use_container_width=True):
            create_project_dialog()
            
        st.markdown("---")
        
        st.subheader("Chat History")
        convs = get_all_conversations(project_id=st.session_state.get("project_id") if st.session_state.get("project_id") != "None" else None)
        if not convs:
            st.write("No previous chats.")
        else:
            for c in convs:
                title = c.get("first_message") or "Empty Chat"
                if len(title) > 30:
                    title = title[:27] + "..."
                col1, col2 = st.columns([5, 1], gap="small")
                with col1:
                    if st.button(f"💬 {title}", key=f"conv_{c['id']}", use_container_width=True):
                        load_conversation(c['id'], c['session_token'])
                with col2:
                    with st.popover("⋮"):
                        if st.button("🗑️ Delete", key=f"del_{c['id']}", use_container_width=True):
                            delete_conversation(c['id'])
                            if st.session_state.get("conversation_db_id") == c['id']:
                                st.session_state.messages = []
                                if "conversation_db_id" in st.session_state:
                                    del st.session_state.conversation_db_id
                            st.rerun()
                            
                        st.markdown("---")
                        st.markdown("**Share Chat**")
                        share_conv_url = f"?share_conv_id={c['id']}"
                        import streamlit.components.v1 as components
                        
                        if st.button("🔗 Copy Link", key=f"share_conv_link_{c['id']}", use_container_width=True):
                            js = f"""
                            <script>
                            const url = window.parent.location.origin + window.parent.location.pathname + "{share_conv_url}";
                            try {{
                                window.parent.navigator.clipboard.writeText(url);
                            }} catch (err) {{
                                const textArea = window.parent.document.createElement("textarea");
                                textArea.value = url;
                                window.parent.document.body.appendChild(textArea);
                                textArea.select();
                                window.parent.document.execCommand('copy');
                                window.parent.document.body.removeChild(textArea);
                            }}
                            </script>
                            """
                            components.html(js, height=0, width=0)
                            st.toast("Chat link copied!", icon="🔗")

        st.markdown("---")
        
        metrics = get_db_metrics()
        
        st.subheader("Knowledge Base Status")
        if metrics:
            st.write(f"**Total Documents:** {metrics['total_docs']}")
            st.write(f"**Total Chunks:** {metrics['total_chunks']}")
            st.write(f"**Total Images:** {metrics['total_images']}")
            st.write(f"**Embedding Model:** `{EMBEDDING_MODEL}`")
            
            st.markdown("---")
            st.subheader("Document Pipeline")
            st.write(f"🟢 **Completed:** {metrics['completed_docs']}")
            st.write(f"🔴 **Failed:** {metrics['failed_docs']}")
            st.write(f"⏳ **Pending:** {metrics['pending_docs']}")
        else:
            st.warning("Could not connect to database.")
            
        st.markdown("---")
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
