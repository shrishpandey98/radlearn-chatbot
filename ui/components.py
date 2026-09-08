"""
ui/components.py
────────────────
Reusable Streamlit components for the Chat interface.
"""
import streamlit as st
import re

def format_answer_citations(text: str, msg_id: str) -> str:
    """Format [1] markers as stylized HTML badges that link to source cards."""
    return re.sub(r'\[(\d+)\]', rf'<a href="#source-{msg_id}-\1" class="citation-marker" title="Go to Source \1">[\1]</a>', text)

def render_metrics(confidence: str, hallucination_risk: str):
    """Render the confidence and risk metrics for an answer."""
    html = f"""
    <div class="metric-container">
        <div class="metric-pill">🎯 Confidence: {confidence}</div>
        <div class="metric-pill">⚠️ Hallucination Risk: {hallucination_risk}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_source_card(citation: dict, msg_id: str):
    """Render a clean source card for a validated citation."""
    # Clean up title
    title = citation.get('doc_title')
    if not title or title == "Unknown Document" or title == "Unknown":
        title = citation.get('formatted_citation', citation.get('document_id', 'Source Document'))
    
    # Remove raw IDs and extensions
    title = title.replace('_', ' ').replace('.pdf', '').replace('.docx', '')
    
    page_html = ""
    if citation.get('page_number') and str(citation.get('page_number')) not in ['?', 'None', '']:
        page_html += f"<p style='margin: 4px 0;'><strong>Page:</strong> {citation['page_number']}</p>"
        
    source_url = citation.get('source_url')
    if source_url and str(source_url).startswith('http'):
        page_html += f"<p style='margin: 4px 0; word-break: break-all;'><strong>URL:</strong> <a href='{source_url}' target='_blank' style='color: #60a5fa;'>{source_url}</a></p>"
        
    html = f"""
    <div class="source-card" id="source-{msg_id}-{citation['citation_number']}">
        <h4>[{citation['citation_number']}] {title}</h4>
        {page_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    
    # Resolve file path dynamically so downloads work across any deployment directory
    from radlearn.config import DOCUMENTS_DIR
    import os
    from pathlib import Path

    file_path = citation.get('file_path')
    resolved_file = None

    if file_path and os.path.isfile(file_path):
        resolved_file = Path(file_path)
    else:
        doc_id = citation.get('document_id')
        if doc_id:
            candidate_dir = DOCUMENTS_DIR / str(doc_id)
            if candidate_dir.is_dir():
                found_files = list(candidate_dir.glob("*.pdf")) + list(candidate_dir.glob("*.docx")) + list(candidate_dir.glob("*.*"))
                if found_files:
                    resolved_file = found_files[0]
        if not resolved_file and file_path:
            base_name = os.path.basename(file_path)
            candidate = DOCUMENTS_DIR / base_name
            if candidate.is_file():
                resolved_file = candidate
            else:
                matched = list(DOCUMENTS_DIR.rglob(base_name))
                if matched:
                    resolved_file = matched[0]

    if resolved_file and resolved_file.exists():
        try:
            with open(resolved_file, "rb") as f:
                file_bytes = f.read()
            st.download_button(
                label=f"📥 Download Source Document",
                data=file_bytes,
                file_name=resolved_file.name,
                mime="application/pdf" if resolved_file.suffix.lower() == ".pdf" else "application/octet-stream",
                key=f"dl_{msg_id}_{citation.get('chunk_id', '')}_{citation['citation_number']}"
            )
        except Exception as e:
            st.caption(f"Document unavailable: {e}")


def render_debug_chunks(chunks: list):
    """Render the raw retrieved chunks for debugging."""
    for i, chunk in enumerate(chunks):
        rank = i + 1
        html = f"""
        <div class="chunk-card">
            <h5>Rank #{rank} — {chunk.get('doc_title', 'Unknown Document')} (Page {chunk.get('page_number', '?')})</h5>
            <div class="score-badge">RRF Score: {chunk.get('rrf_score', 0):.4f}</div>
            <div class="score-badge">Semantic Sim: {chunk.get('similarity_score', 0):.4f}</div>
            <p style="margin-top: 8px; font-size: 0.85rem; color: #666;">{chunk.get('text', '')[:250]}...</p>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

def render_debug_docs(chunks: list):
    """Render the unique documents retrieved."""
    # Extract unique documents
    unique_docs = []
    seen = set()
    for chunk in chunks:
        doc_id = chunk.get("document_id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(chunk)
            
    if not unique_docs:
        st.write("No documents retrieved.")
        return
        
    for doc in unique_docs:
        html = f"""
        <div class="chunk-card">
            <h5>{doc.get('doc_title', 'Unknown Document')}</h5>
            <p style="margin-bottom:0; font-size: 0.85rem; color: #666;">ID: {doc.get('document_id', 'Unknown')}</p>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

def render_citation_pills(citations: list, msg_id: str):
    """Render citations as sleek horizontal pills."""
    if not citations:
        return
        
    html = '<div class="citation-pills-container">'
    for cit in citations:
        doc_title = cit.get('formatted_citation', cit.get('document_id', 'Source Document'))
        html += f'<a href="#source-{msg_id}-{cit["citation_number"]}" class="citation-pill">📖 [{cit["citation_number"]}] {doc_title}</a>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_structured_answer(data: dict, citations: list, msg_id: str):
    """Render the structured JSON clinical answer."""
    
    # 1. Summary
    if data.get("summary"):
        html = f"""
        <div class="struct-card">
            <div class="struct-header">✨ SUMMARY</div>
            <p style="margin: 0; color: #e2e8f0; line-height: 1.6;">{data['summary']}</p>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        
    # 2. Key Findings
    if data.get("key_findings"):
        st.markdown("""<div class="struct-card" style="padding-bottom: 0;"><div class="struct-header">📄 KEY MRI FINDINGS</div>""", unsafe_allow_html=True)
        for finding in data["key_findings"]:
            with st.expander(f"✅ {finding.get('title', 'Finding')}"):
                st.markdown(finding.get('details', ''))
        st.markdown("</div>", unsafe_allow_html=True)
        
    # 3. Differential Diagnosis
    if data.get("differential_diagnosis"):
        html = """
        <div class="struct-card">
            <div class="struct-header">🧠 DIFFERENTIAL DIAGNOSIS</div>
        """
        for diff in data["differential_diagnosis"]:
            prob = diff.get("probability", "LOW").upper()
            badge_class = "badge-high" if prob == "HIGH" else "badge-medium" if prob == "MEDIUM" else "badge-low"
            
            html += f"""
            <div class="diff-row">
                <div class="diff-header-row">
                    <span class="prob-badge {badge_class}">{prob}</span>
                    <span class="cond-title">{diff.get('condition', 'Condition')}</span>
                </div>
                <div class="cond-details">{diff.get('details', '')}</div>
            </div>
            """
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
        
    # 4. Teaching Points
    if data.get("teaching_points"):
        html = """
        <div class="struct-card">
            <div class="struct-header">⭐ TEACHING POINTS</div>
        """
        for i, point in enumerate(data["teaching_points"]):
            html += f"""
            <div class="teaching-point">
                <div class="tp-number">{i+1}</div>
                <div class="tp-text">{point}</div>
            </div>
            """
        
        # Embed citation pills directly in teaching points card to match screenshot
        if citations:
            html += '<div class="citation-pills-container">'
            for cit in citations:
                doc_title = cit.get('formatted_citation', cit.get('document_id', 'Source Document'))
                html += f'<a href="#source-{msg_id}-{cit["citation_number"]}" class="citation-pill">📖 [{cit["citation_number"]}] {doc_title}</a>'
            html += '</div>'
            
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
        
    # 5. Follow-up Questions
    if data.get("follow_up_questions"):
        st.markdown("<p style='color: #94a3b8; font-size: 0.9rem; margin-top: 24px; margin-bottom: 8px;'>Suggested follow-up questions</p>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(data["follow_up_questions"]):
            if cols[i % 2].button(q, key=f"followup_{msg_id}_{i}"):
                st.session_state.chat_input_override = q
                st.rerun()
