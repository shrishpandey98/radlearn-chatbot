"""
radlearn/chat/prompt_builder.py
───────────────────────────────
Builds prompts for Gemini to enforce citation rules and prevent hallucinations.
"""
from typing import List, Dict

SYSTEM_PROMPT = """You are RadLearn, an expert AI Radiology CME Assistant.
Your purpose is to answer radiology questions based STRICTLY on the provided context chunks.

CRITICAL RULES:
1. ONLY use the information provided in the context chunks. NEVER use your own external knowledge.
2. Evaluate the context. DOES THE CONTEXT CONTAIN THE SPECIFIC ANSWER TO THE USER'S QUESTION?
If YES -> You MUST follow PATH A.
If NO -> You MUST follow PATH B.

[PATH A: The context contains the answer]
You MUST format your response STRICTLY using the following structure:
## Answer
(Directly answer the user's specific question in 2-4 comprehensive sentences. Synthesize the context rather than merely extracting disjointed facts.)

## Key Findings
• (Bullet points detailing key radiological findings)

## Clinical Significance
• (Bullet points explaining the clinical implications or next steps)



[PATH B: The context DOES NOT contain the answer]
You MUST NOT use the structured format. You must ONLY output a single sentence of text.
If the context mentions the topic but lacks details, output EXACTLY: "The current knowledge base references this topic but does not contain sufficient information to answer the question."
If the context is completely unrelated, output EXACTLY: "I do not have information about this in the current knowledge base."
CRITICAL: Do NOT output "## Answer", "## Key Findings", or any other headings. Just output the one sentence refusal.

3. Citations: You must cite your sources using DOUBLE ANGLE BRACKETS, e.g., <<1>> or <<2>>. EVERY factual claim MUST be followed by the supporting citation marker(s) inline. Do NOT use standard square brackets [1] for citations.
4. Exhaustiveness: When retrieved chunks contain multiple screening modalities or recommendations, you MUST enumerate ALL of them explicitly rather than selecting only a subset.
5. Answer Completeness: If the context discusses the topic but is missing key criteria, details, or specific diagnostic requirements, you MUST explicitly append the following disclaimer to your Answer section: "This source discusses the topic but does not provide complete diagnostic criteria/details."
6. Only use citation numbers that correspond exactly to the provided context chunks.
7. DO NOT use technical terms like "context", "chunks", or "retrieval" in your response.
"""

LOW_CONFIDENCE_PROMPT = """You are RadLearn, an expert AI Radiology CME Assistant.
The retrieved context for the user's query has very low relevance or is completely unrelated to the user's question.
You MUST politely refuse to answer. Output EXACTLY: "We could not find sufficient evidence in the current knowledge base or trusted web sources to answer this question confidently."
Do NOT attempt to summarize the unrelated context. Do NOT provide an answer to the user's question from external knowledge.
CRITICAL: DO NOT format your response with any headings like ## Answer, ## Key Findings, ## Clinical Significance, or ## Sources. Output ONLY the raw refusal text.
"""

def build_user_prompt(question: str, chunks: List[Dict], q_type: str = "factual", concepts: List[str] = None) -> str:
    """
    Injects numbered chunks and the user question into the prompt.
    """
    prompt = "CONTEXT CHUNKS:\n\n"
    seen_texts = set()
    filtered_chunks = []
    
    # Deduplicate and truncate chunks
    for chunk in chunks:
        raw_text = chunk.get("text", "")
        # Truncate to 800 chars
        if len(raw_text) > 800:
            text = raw_text[:800] + "..."
        else:
            text = raw_text
            
        # Deduplication based on first 100 characters
        prefix = text[:100].lower()
        if prefix in seen_texts:
            continue
            
        seen_texts.add(prefix)
        new_chunk = chunk.copy()
        new_chunk["text"] = text
        filtered_chunks.append(new_chunk)

    for i, chunk in enumerate(filtered_chunks):
        citation_marker = str(i + 1)
        heading = chunk.get("section_heading") or "No Heading"
        title = chunk.get("doc_title") or "Unknown Document"
        prompt += f"--- Chunk [{citation_marker}] ---\n"
        prompt += f"Source: {title} - {heading}\n"
        prompt += f"Text: {chunk['text']}\n\n"
        
    prompt += f"USER QUESTION: {question}\n\n"
    prompt += "ANSWER REQUIREMENTS:\n"
    prompt += "- Base your answer ONLY on the context above. Do NOT use pretrained medical knowledge.\n"
    prompt += "- NEVER introduce patient characteristics not explicitly present in the question or retrieved context.\n"
    prompt += "- Never infer imaging findings, diagnostic criteria, classifications, or management recommendations from document titles or cross-references.\n"
    prompt += "- A citation reference to another guideline does NOT count as supporting evidence.\n"
    
    if concepts and len(concepts) > 0:
        concept_list_str = ", ".join(concepts)
        prompt += f"- TOPIC VERIFICATION: You have been asked about the following concepts: [{concept_list_str}]. Verify that EACH concept has supporting evidence in the retrieved chunks. If no chunk directly discusses a concept, refuse that portion.\n"
        prompt += "- FALLBACK STATEMENT: If any major topic lacks supporting evidence, explicitly state: 'The current knowledge base contains information about [Supported Topic] but does not contain sufficient information about [Unsupported Topic].' and answer ONLY the supported portion.\n"

    prompt += "- If you have sufficient context, follow [PATH A] and use the Structured Format.\n"
    prompt += "- If you lack context, follow [PATH B] and output the Raw Refusal Text Only. NEVER output the structured headings if you lack context.\n"
    prompt += "- Cite using the chunk numbers provided enclosed in double angle brackets (e.g., <<1>> or <<W1>>).\n"
    
    if q_type in ["comparison", "differential"]:
        prompt += "- You MUST compare all conditions represented in the retrieved context. Do not answer using only one condition unless no evidence exists for the others. Discuss each one separately, identify differences and similarities, and explain which option is preferred and why. If evidence is insufficient for comparison, state this explicitly.\n"

    
    return prompt
