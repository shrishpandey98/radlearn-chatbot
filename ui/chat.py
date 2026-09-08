"""
ui/chat.py
──────────
Main ChatGPT-style chat interface logic.
"""
import streamlit as st
import time
from radlearn.chat.engine import process_query
from ui.components import (
    format_answer_citations,
    render_metrics,
    render_source_card,
    render_debug_chunks,
    render_debug_docs
)
from radlearn.database.conversations import create_conversation, create_message, get_next_message_index, update_message_feedback
from radlearn.database.citations import batch_insert_citations

def render_feedback_buttons(msg: dict):
    if "id" not in msg:
        return
        
    up_type = "primary" if msg.get("was_helpful") is True else "secondary"
    down_type = "primary" if msg.get("was_helpful") is False else "secondary"

    col1, col2, col3, col4, _ = st.columns([1, 1, 1, 1, 10])
    with col1:
        if st.button("👍", help="Helpful", key=f"up_{msg['id']}", type=up_type):
            if msg.get("was_helpful") is not True:
                update_message_feedback(msg["id"], 5, True)
                msg["was_helpful"] = True
                st.rerun()
    with col2:
        if st.button("👎", help="Not Helpful", key=f"down_{msg['id']}", type=down_type):
            if msg.get("was_helpful") is not False:
                update_message_feedback(msg["id"], 1, False)
                msg["was_helpful"] = False
                st.rerun()
    with col3:
        if st.button("📋", help="Copy response", key=f"copy_{msg['id']}"):
            import json
            import streamlit.components.v1 as components
            text_to_copy = json.dumps(msg["content"])
            copy_js = f"""
            <script>
            const text = {text_to_copy};
            try {{
                window.parent.navigator.clipboard.writeText(text);
            }} catch (err) {{
                const textArea = window.parent.document.createElement("textarea");
                textArea.value = text;
                window.parent.document.body.appendChild(textArea);
                textArea.select();
                window.parent.document.execCommand('copy');
                window.parent.document.body.removeChild(textArea);
            }}
            </script>
            """
            components.html(copy_js, height=0, width=0)
            st.toast("Copied to clipboard!", icon="📋")
    
    with col4:
        with st.popover("📤"):
            st.markdown("**Share this response**")
            share_url = f"?share_msg_id={msg['id']}"
            import streamlit.components.v1 as components
            
            if st.button("🔗 Copy Link", key=f"share_link_{msg['id']}"):
                js = f"""
                <script>
                const url = window.parent.location.origin + window.parent.location.pathname + "{share_url}";
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
                st.toast("Link copied to clipboard!", icon="🔗")


def hydrate_message_from_db(db_msg: dict) -> dict:
    if db_msg["role"] == "user":
        return {
            "id": db_msg["id"],
            "role": "user",
            "content": db_msg["content"]
        }
    
    from radlearn.database.citations import get_citations_for_message
    citations = get_citations_for_message(db_msg["id"])
    
    score = db_msg.get("top_similarity_score", 0.0)
    if score is None: score = 0.0
    band = "🟢 High Confidence" if score > 0.75 else "🟡 Medium Confidence" if score > 0.6 else "🔴 Low Confidence"
    
    return {
        "id": db_msg["id"],
        "role": "assistant",
        "content": db_msg["content"],
        "citations": citations,
        "retrieved_chunks": [], 
        "confidence": score,
        "confidence_band": band,
        "hallucination_risk": "UNKNOWN",
        "latency": db_msg.get("total_time_ms", 0) / 1000.0 if db_msg.get("total_time_ms") else 0,
        "is_error": False,
        "was_helpful": db_msg.get("was_helpful") == 1 if db_msg.get("was_helpful") in [0, 1] else None
    }


def render_chat():
    import uuid
    from radlearn.ingestion.pipeline import ingest_file
    from radlearn.database.documents import delete_document_record

    params = st.query_params
    
    if "share_msg_id" in params and "shared_loaded" not in st.session_state:
        from radlearn.database.conversations import get_message, get_message_by_index
        msg_id = params["share_msg_id"]
        asst_msg = get_message(msg_id)
        if asst_msg and asst_msg["role"] == "assistant":
            user_msg = get_message_by_index(asst_msg["conversation_id"], asst_msg["message_index"] - 1)
            st.session_state.messages = []
            if user_msg:
                st.session_state.messages.append(hydrate_message_from_db(user_msg))
            st.session_state.messages.append(hydrate_message_from_db(asst_msg))
            st.session_state.shared_loaded = True
            
    elif "share_conv_id" in params and "shared_loaded" not in st.session_state:
        from radlearn.database.conversations import get_conversation_messages
        conv_id = params["share_conv_id"]
        msgs = get_conversation_messages(conv_id)
        st.session_state.messages = [hydrate_message_from_db(m) for m in msgs]
        st.session_state.shared_loaded = True

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "session_uploaded_documents" not in st.session_state:
        st.session_state.session_uploaded_documents = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []

    welcome_container = st.empty()
    if not st.session_state.messages:
        welcome_container.markdown("""<div style='display: flex; flex-direction: column; align-items: center; margin-top: 4vh;'>
<div style='text-align: center; max-width: 600px;'>
<h1 style='font-weight: 800; letter-spacing: 1px; color: #ffffff; margin-bottom: 0;'>RadLearn</h1>
<h3 style='color: #94a3b8; font-weight: 400; margin-top: 5px; margin-bottom: 20px;'>AI-powered Radiology CME Assistant</h3>
<div style='display: inline-block; text-align: left; color: #e2e8f0; font-size: 1rem; line-height: 1.4;'>
<p style='margin: 4px 0;'>✓ Ask questions from radiology guidelines</p>
<p style='margin: 4px 0;'>✓ Get source-backed answers</p>
<p style='margin: 4px 0;'>✓ Upload your own documents</p>
<p style='margin: 4px 0;'>✓ View citations and source material</p>
</div>
<p style='color: #64748b; font-size: 0.75rem; margin-top: 25px; line-height: 1.3;'><strong>Educational use only.</strong><br>Responses are generated from uploaded educational materials and should not be used as the sole basis for clinical decision making.</p>
</div>
</div>""", unsafe_allow_html=True)

    # Display chat history
    for idx, msg in enumerate(st.session_state.messages):
        msg_id = str(idx)
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                if msg.get("is_error"):
                    st.markdown(f"🚫 **{msg['content']}**")
                else:
                    formatted_ans = format_answer_citations(msg["content"], msg_id)
                    st.markdown(formatted_ans, unsafe_allow_html=True)
                    
                    render_metrics(msg.get("confidence_band", msg.get("confidence", "Unknown")), msg["hallucination_risk"])
                    
                    render_feedback_buttons(msg)
                    

                    if msg.get("citations"):
                        st.markdown("### 📚 Source Documents")
                        for c in msg["citations"]:
                            render_source_card(c, msg_id)
                            
                    if msg.get("retrieved_chunks"):
                        with st.expander("🛠️ Debug: Retrieved Chunks", expanded=False):
                            render_debug_chunks(msg["retrieved_chunks"])
                        
                        with st.expander("🛠️ Debug: Retrieved Documents", expanded=False):
                            render_debug_docs(msg["retrieved_chunks"])

    # Active Documents & File Upload Control
    if st.session_state.session_uploaded_documents:
        with st.container():
            st.markdown("### 📎 Active Documents")
            for doc_id, doc in list(st.session_state.session_uploaded_documents.items()):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"✓ **{doc['filename']}** (Pages: {doc['pages']} | Chunks: {doc['chunks']})")
                with col2:
                    if st.button("Remove", key=f"rm_{doc_id}"):
                        delete_document_record(doc_id)
                        del st.session_state.session_uploaded_documents[doc_id]
                        st.rerun()

    with st.popover("📎"):
        uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file to chat with it.", type=["pdf", "docx", "txt"])
        if uploaded_file:
            with st.status("Uploading...") as status:
                st.write("Parsing document...")
                start_time = time.time()
                file_bytes = uploaded_file.read()
                filename = uploaded_file.name
                metadata = {
                    "title": filename,
                    "doc_type": "article",
                    "source_url": "session_upload",
                    "source_type": "user_upload",
                    "session_id": st.session_state.session_id,
                    "knowledge_source_id": st.session_state.session_id,
                    "project_id": st.session_state.get("project_id") if st.session_state.get("project_id") != "None" else None
                }
                try:
                    st.write("Creating chunks & generating embeddings...")
                    res = ingest_file(file_bytes, filename, metadata)
                    processing_time = round(time.time() - start_time, 2)
                    
                    if res.get("status") == "success":
                        st.session_state.session_uploaded_documents[res["doc_id"]] = {
                            "document_id": res["doc_id"],
                            "filename": filename,
                            "pages": res.get("pages", 0),
                            "chunks": res.get("chunk_count", 0),
                            "upload_time": time.time()
                        }
                        status.update(label="Complete.", state="complete", expanded=False)
                        st.success(f"Document uploaded successfully.\n\nFile Name: {filename}\nPages: {res.get('pages', 0)}\nChunks: {res.get('chunk_count', 0)}\nImages: {res.get('image_count', 0)}\nProcessing Time: {processing_time}s")
                        st.rerun()
                    elif res.get("status") == "skipped":
                        status.update(label="File already exists.", state="complete", expanded=False)
                        st.warning("File is already in the knowledge base.")
                    else:
                        status.update(label="Ingestion failed.", state="error")
                        st.error(res.get("reason", "Unknown error."))
                except Exception as e:
                    status.update(label="Ingestion failed.", state="error")
                    st.error(f"Error: {str(e)}")

    # Chat Input
    if prompt := st.chat_input("Ask a clinical radiology question..."):
        welcome_container.empty()
        
        if "conversation_db_id" not in st.session_state:
            current_project_id = st.session_state.get("project_id") if st.session_state.get("project_id") != "None" else None
            conv = create_conversation(session_token=st.session_state.session_id, project_id=current_project_id)
            st.session_state.conversation_db_id = conv["id"]
        
        conv_db_id = st.session_state.conversation_db_id
        user_msg_idx = get_next_message_index(conv_db_id)
        create_message(
            conversation_id=conv_db_id,
            role="user",
            content=prompt,
            message_index=user_msg_idx
        )

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            try:
                start_time = time.time()
                with st.spinner("Searching knowledge base..."):
                    from radlearn.chat.engine import prepare_query, finalize_answer
                    from radlearn.chat.llm_factory import get_llm_client
                    current_project_id = st.session_state.get("project_id") if st.session_state.get("project_id") != "None" else None
                    prep_res = prepare_query(prompt, session_id=st.session_state.session_id, project_id=current_project_id)
                
                latency = time.time() - start_time
                
                if prep_res.get("status") == "quota_exceeded":
                    error_msg = "Something went wrong, try again later."
                    st.markdown(f"🚫 **{error_msg}**")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "is_error": True})
                elif prep_res.get("status") == "invalid_api_key":
                    error_msg = "The Google API key is invalid. Please check your .env file."
                    st.markdown(f"🚫 **{error_msg}**")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "is_error": True})
                elif not prep_res.get("ranked_chunks"):
                    error_msg = "No relevant information found in the knowledge base."
                    st.markdown(f"🚫 **{error_msg}**")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "is_error": True})
                else:
                    t2 = time.time()
                    client = get_llm_client()
                    raw_answer = ""
                    placeholder = st.empty()
                    
                    try:
                        stream_gen = client.generate_stream(prep_res["sys_prompt"], prep_res["user_prompt"])
                        with placeholder:
                            raw_answer = st.write_stream(stream_gen)
                    except Exception as stream_err:
                        raw_answer = ""

                    if not raw_answer:
                        try:
                            raw_answer = client.generate_answer(prep_res["sys_prompt"], prep_res["user_prompt"])
                            placeholder.markdown(raw_answer)
                        except Exception as gen_err:
                            raw_answer = f"Error generating answer: {gen_err}"
                            placeholder.markdown(raw_answer)

                    llm_gen_ms = int((time.time() - t2) * 1000)
                        
                    final_res = finalize_answer(raw_answer, prep_res["ranked_chunks"])

                    
                    # The new message index will be len(st.session_state.messages) - 1 since we already appended the user prompt
                    msg_id = str(len(st.session_state.messages))
                    formatted_ans = format_answer_citations(final_res["answer"], msg_id)
                    placeholder.markdown(formatted_ans, unsafe_allow_html=True)
                    
                    # Reconstruct `res` dict for the downstream metrics and DB saving logic
                    res = {
                        "answer": final_res["answer"],
                        "citations": final_res["citations"],
                        "retrieved_chunks": prep_res["ranked_chunks"],
                        "confidence_score": prep_res["confidence_score"],
                        "confidence_band": prep_res["confidence_band"],
                        "expanded_query": prep_res["expanded_query"]
                    }
                        
                    # Assuming risk evaluation isn't strictly required to be re-run here, but we will grab it if it's not present,
                    # Actually process_query does not return hallucination_risk natively, it's evaluated later in old app.py.
                    # Wait, in the old app.py we called evaluator.py. Since the user asked not to change the architecture, 
                    # I'll just use a mock or call evaluator if it was there. But the user said "Use the existing architecture".
                    # Let's import calculate_grounding_metrics
                    from radlearn.chat.evaluator import calculate_grounding_metrics
                    eval_metrics = calculate_grounding_metrics(
                        answer_text=res["answer"],
                        citations=res["citations"],
                        retrieved_chunks=res["retrieved_chunks"],
                        confidence_score=res["confidence_score"],
                        question=prompt
                    )
                    risk_level = eval_metrics["hallucination_risk"]
                    
                    # 1. Hallucination Risk Scoring Adjustments
                    if risk_level == "HIGH":
                        res["confidence_band"] = "🔴 Low Confidence"
                    
                    if not prep_res.get("ranked_chunks"):
                        res["confidence_band"] = "🔴 Low Confidence"
                        
                    if prep_res.get("web_search_triggered"):
                        # If web search was used, confidence can be at most MEDIUM
                        if res["confidence_band"] == "🟢 High Confidence":
                            res["confidence_band"] = "🟡 Medium Confidence"
                            
                    render_metrics(res["confidence_band"], risk_level)
                    
                    if res["citations"]:
                        st.markdown("### 📚 Source Documents")
                        
                        # 5. Source Attribution
                        source_text = "✓ Knowledge Base + 🌐 Web Search" if prep_res.get("web_search_triggered") else "✓ Knowledge Base"
                        if not res["citations"] and prep_res.get("web_search_triggered"):
                            source_text = "🌐 Web Search"
                        
                        st.markdown(f"**Source Type:**\n{source_text}")
                        
                        for c in res["citations"]:
                            render_source_card(c, msg_id)
                            
                    if res["retrieved_chunks"]:
                        with st.expander("⏱️ Debug: Latency Breakdown", expanded=False):
                            retrieval_ms = prep_res.get("retrieval_ms", 0)
                            prompt_build_ms = prep_res.get("prompt_build_ms", 0)
                            prompt_tokens = prep_res.get("prompt_tokens", 0)
                            web_search_triggered = prep_res.get("web_search_triggered", False)
                            web_search_ms = prep_res.get("web_search_ms", 0)
                            web_domains = prep_res.get("web_domains_used", [])
                            
                            total_ms = retrieval_ms + prompt_build_ms + llm_gen_ms + web_search_ms
                            
                            retrieval_quality = prep_res.get("retrieval_quality", "UNKNOWN")
                            avg_similarity = prep_res.get("avg_similarity", 0.0)
                            keyword_matches = prep_res.get("keyword_matches", 0)
                            
                            st.markdown(f"""
                            - **Retrieval:** {retrieval_ms} ms
                            - **PubMed Search Fallback:** {web_search_ms} ms {'(Triggered)' if web_search_triggered else '(Not Triggered)'}
                            - **Prompt Build:** {prompt_build_ms} ms
                            - **LLM Generation:** {llm_gen_ms} ms
                            - **Total:** {total_ms} ms
                            
                            - **Retrieval Quality:** {retrieval_quality}
                            - **Average Similarity:** {avg_similarity:.4f}
                            - **Keyword Matches:** {keyword_matches}
                            - **Prompt Tokens (est.):** ~{prompt_tokens}
                            """)
                            if web_search_triggered:
                                st.markdown(f"**Web Domains Searched:** {', '.join(web_domains)}")
                            
                        with st.expander("🛠️ Debug: Retrieved Chunks", expanded=False):
                            render_debug_chunks(res["retrieved_chunks"])
                        
                        with st.expander("🛠️ Debug: Retrieved Documents", expanded=False):
                            render_debug_docs(res["retrieved_chunks"])
                            
                    asst_msg_idx = get_next_message_index(conv_db_id)
                    db_msg = create_message(
                        conversation_id=conv_db_id,
                        role="assistant",
                        content=res["answer"],
                        message_index=asst_msg_idx,
                        expanded_query=res.get("expanded_query", ""),
                        top_similarity_score=res["confidence_score"],
                        chunks_retrieved=len(res.get("retrieved_chunks", [])),
                        total_time_ms=int(latency * 1000)
                    )
                    if res.get("citations"):
                        batch_insert_citations(res["citations"], db_msg["id"])

                    new_msg_state = {
                        "id": db_msg["id"],
                        "role": "assistant",
                        "content": res["answer"],
                        "citations": res["citations"],
                        "retrieved_chunks": res["retrieved_chunks"],
                        "confidence": res["confidence_score"],
                        "confidence_band": res["confidence_band"],
                        "hallucination_risk": risk_level,
                        "latency": latency,
                        "is_error": False,
                        "was_helpful": None
                    }
                    # Save state
                    st.session_state.messages.append(new_msg_state)
                    
                    render_feedback_buttons(new_msg_state)
                    
                    import streamlit.components.v1 as components
                    scroll_js = f"""
                    <script>
                        // Force re-render for new message: {msg_id}
                        function scrollToBottom() {{
                            var messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
                            if (messages && messages.length > 0) {{
                                messages[messages.length - 1].scrollIntoView({{behavior: 'smooth', block: 'end'}});
                            }}
                        }}
                        scrollToBottom();
                        setTimeout(scrollToBottom, 200);
                        setTimeout(scrollToBottom, 600);
                    </script>
                    """
                    components.html(scroll_js, height=0)
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                st.markdown(f"🚫 **{error_msg}**")
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "is_error": True})
