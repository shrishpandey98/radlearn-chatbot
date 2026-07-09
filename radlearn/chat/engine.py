"""
radlearn/chat/engine.py
────────────────────────
Main orchestration layer.
Question -> Preprocess -> Embed -> Hybrid Retrieval -> RRF -> Prompt -> LLM -> Citations.
"""
from radlearn.retrieval.preprocessor import preprocess_query
from radlearn.retrieval.searcher import hybrid_search
from radlearn.retrieval.ranker import reciprocal_rank_fusion
from radlearn.chat.prompt_builder import SYSTEM_PROMPT, LOW_CONFIDENCE_PROMPT, build_user_prompt
from radlearn.chat.llm_factory import get_llm_client
from radlearn.chat.citation_parser import parse_citations
from radlearn.retrieval.classifier import classify_query
import re
import time

def execute_retrieval(question: str, expanded_query: str, eff_specialty: str, session_id: str, project_id: str = None) -> dict:
    classification = classify_query(question)
    q_type = classification["type"]
    concepts = classification["concepts"]

    all_semantic = []
    all_keyword = []
    
    for concept in concepts:
        concept_prep = preprocess_query(concept)
        concept_expanded = concept_prep["expanded_query"]
        # Reduce top_k per concept to fit within limits
        res = hybrid_search(concept, concept_expanded, specialty_filter=eff_specialty, top_k=5, session_id=session_id, project_id=project_id)
        all_semantic.append(res["semantic_results"])
        all_keyword.append(res["keyword_results"])
        
    return {
        "semantic_results": all_semantic,
        "keyword_results": all_keyword,
        "q_type": q_type,
        "concepts": concepts
    }


def prepare_query(question: str, session_id: str = None, specialty_filter: str = None, project_id: str = None) -> dict:
    """
    Setup the query context before making an LLM call. Used for streaming.
    """
    prep_res = preprocess_query(question)
    expanded_query = prep_res["expanded_query"]
    
    eff_specialty = specialty_filter or prep_res["specialty"]
    if eff_specialty == "general":
        eff_specialty = None
        
    t0 = time.time()
    try:
        search_res = execute_retrieval(question, expanded_query, eff_specialty, session_id, project_id)
        q_type = search_res["q_type"]
        concepts = search_res["concepts"]
    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
            return {
                "status": "quota_exceeded",
                "message": "Something went wrong, try again later.",
                "error_details": str(e)
            }
        elif "400" in error_str or "401" in error_str or "api key not valid" in error_str or "api_key_invalid" in error_str or "authentication" in error_str:
            return {
                "status": "invalid_api_key",
                "message": "The Google API key is invalid. Please check your .env file.",
                "error_details": str(e)
            }
        raise e
    
    top_n = 6 if q_type in ["comparison", "differential"] and len(concepts) > 1 else 5
    ranked_chunks = reciprocal_rank_fusion(
        semantic_results_list=search_res["semantic_results"],
        keyword_results_list=search_res["keyword_results"],
        top_n=top_n
    )
    retrieval_ms = int((time.time() - t0) * 1000)
    
    confidence_score = ranked_chunks[0]["rrf_score"] if ranked_chunks else 0.0
    
    max_sim = max([c.get("similarity_score", 0.0) for c in ranked_chunks]) if ranked_chunks else 0.0
    
    avg_similarity = sum(c.get("similarity_score", 0.0) for c in ranked_chunks) / len(ranked_chunks) if ranked_chunks else 0.0
    
    keyword_matches = 0
    if ranked_chunks:
        combined_text = " ".join([c.get("text", "").lower() for c in ranked_chunks])
        for concept in concepts:
            if concept.lower() in combined_text:
                keyword_matches += 1
                
    retrieval_quality = "HIGH"
    if avg_similarity < 0.85 or keyword_matches < 1:
        retrieval_quality = "LOW"
        
    web_search_triggered = False
    web_domains_used = []
    web_search_ms = 0
    
    if not ranked_chunks or retrieval_quality == "LOW" or max_sim < 0.82 or confidence_score < 0.02:
        from radlearn.retrieval.web_search import perform_web_search
        t_web = time.time()
        
        # Use extracted concepts for PubMed as it chokes on natural language words like "What is"
        search_term = " ".join(concepts) if concepts else question
        web_chunks = perform_web_search(search_term)
        
        if web_chunks:
            web_search_triggered = True
            web_domains_used = ["PubMed"]
            
            # Since web chunks are highly relevant fallback, prepend them to ranked_chunks
            # We assign them a synthetic RRF and similarity score to guarantee they appear in the LLM prompt
            # and prevent the LOW_CONFIDENCE_PROMPT from being used.
            for i, wc in enumerate(web_chunks):
                wc["rrf_score"] = 1.0 - (i * 0.01)
                wc["similarity_score"] = 0.99
                
            ranked_chunks = (web_chunks + ranked_chunks)[:top_n]
            
            confidence_score = ranked_chunks[0]["rrf_score"] if ranked_chunks else 0.0
            max_sim = max([c.get("similarity_score", 0.0) for c in ranked_chunks]) if ranked_chunks else 0.0
        web_search_ms = int((time.time() - t_web) * 1000)

    if max_sim > 0.84 and len(ranked_chunks) >= 2:
        confidence_band = "🟢 High Confidence"
    elif max_sim >= 0.82:
        confidence_band = "🟡 Medium Confidence"
    else:
        confidence_band = "🔴 Low Confidence"
        
    if not ranked_chunks or (max_sim < 0.82 and not web_search_triggered):
        sys_prompt = LOW_CONFIDENCE_PROMPT
    else:
        sys_prompt = SYSTEM_PROMPT
        if project_id:
            from radlearn.database.projects import get_project
            project = get_project(project_id)
            if project and project.get("system_instructions"):
                # Append the custom instructions to the base system prompt
                sys_prompt = sys_prompt + "\n\nPROJECT CUSTOM INSTRUCTIONS:\n" + project["system_instructions"]
        
    t1 = time.time()
    user_prompt = build_user_prompt(question, ranked_chunks, q_type=q_type, concepts=concepts)
    prompt_build_ms = int((time.time() - t1) * 1000)
    prompt_tokens = len(sys_prompt + user_prompt) // 4
    
    return {
        "status": "ready",
        "sys_prompt": sys_prompt,
        "user_prompt": user_prompt,
        "ranked_chunks": ranked_chunks,
        "confidence_score": confidence_score,
        "confidence_band": confidence_band,
        "expanded_query": expanded_query,
        "retrieval_ms": retrieval_ms,
        "prompt_build_ms": prompt_build_ms,
        "prompt_tokens": prompt_tokens,
        "web_search_triggered": web_search_triggered,
        "web_domains_used": web_domains_used,
        "web_search_ms": web_search_ms,
        "retrieval_quality": retrieval_quality,
        "avg_similarity": avg_similarity,
        "keyword_matches": keyword_matches
    }

def finalize_answer(raw_answer: str, ranked_chunks: list) -> dict:
    """
    Process raw text streamed from the LLM, format citations and validate them.
    """
    if not ranked_chunks:
        return {
            "answer": "Information not available in knowledge base.",
            "citations": []
        }
        
    REFUSAL_PATTERNS = [
        "does not contain sufficient information",
        "knowledge base does not contain",
        "cannot answer",
        "not available in the provided context",
        "insufficient information",
        "unable to answer",
        "not mentioned in the provided context",
        "do not have information about this",
        "no information is available"
    ]
    
    lower_ans = raw_answer.lower()
    is_refusal = any(pat in lower_ans for pat in REFUSAL_PATTERNS)
    
    if is_refusal:
        return {
            "answer": "We could not find sufficient evidence in the current knowledge base or trusted web sources to answer this question confidently.",
            "citations": []
        }
        
    # 1. Strip out PDF bibliography pollution (any existing [N] brackets)
    answer = re.sub(r'\[[\d\s,\-]+\]', '', raw_answer)
    
    # 2. Normalize grouped RAG citations (e.g., <<2, 4, 5>> -> <<2>> <<4>> <<5>>)
    def _repl(m):
        nums = re.findall(r'\d+', m.group(1))
        return " ".join([f"<<{n}>>" for n in nums])
    answer = re.sub(r'<<([\d\s,]+)>>', _repl, answer)
    
    # 3. Convert unique RAG markers back to standard UI markers (<<N>> -> [N])
    answer = re.sub(r'<<(\d+)>>', r'[\1]', answer)
    
    citations = parse_citations(answer, ranked_chunks)
    
    return {
        "answer": answer,
        "citations": citations
    }


def process_query(question: str, session_id: str = None, specialty_filter: str = None, project_id: str = None) -> dict:
    """
    End-to-end pipeline for answering a user question.
    """
    # 1. Preprocess
    prep_res = preprocess_query(question)
    expanded_query = prep_res["expanded_query"]
    
    # Use detected specialty if not explicitly filtered
    eff_specialty = specialty_filter or prep_res["specialty"]
    if eff_specialty == "general":
        eff_specialty = None # Do not filter if general
        
    t0 = time.time()
    # 2 & 3. Hybrid Retrieval
    try:
        search_res = execute_retrieval(question, expanded_query, eff_specialty, session_id, project_id)
        q_type = search_res["q_type"]
        concepts = search_res.get("concepts", [])
    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
            return {
                "status": "quota_exceeded",
                "message": "Something went wrong, try again later.",
                "error_details": str(e)
            }
        elif "400" in error_str or "401" in error_str or "api key not valid" in error_str or "api_key_invalid" in error_str or "authentication" in error_str:
            return {
                "status": "invalid_api_key",
                "message": "The Google API key is invalid. Please check your .env file.",
                "error_details": str(e)
            }
        raise e
    
    # 4. RRF Ranking
    top_n = 6 if q_type in ["comparison", "differential"] and len(concepts) > 1 else 5
    ranked_chunks = reciprocal_rank_fusion(
        semantic_results_list=search_res["semantic_results"],
        keyword_results_list=search_res["keyword_results"],
        top_n=top_n
    )
    retrieval_ms = int((time.time() - t0) * 1000)
    
    # Calculate confidence (using max rrf_score as proxy)
    confidence_score = ranked_chunks[0]["rrf_score"] if ranked_chunks else 0.0
    
    # Calculate confidence band based on max semantic similarity
    max_sim = max([c.get("similarity_score", 0.0) for c in ranked_chunks]) if ranked_chunks else 0.0
    
    avg_similarity = sum(c.get("similarity_score", 0.0) for c in ranked_chunks) / len(ranked_chunks) if ranked_chunks else 0.0
    
    keyword_matches = 0
    if ranked_chunks:
        combined_text = " ".join([c.get("text", "").lower() for c in ranked_chunks])
        for concept in concepts:
            if concept.lower() in combined_text:
                keyword_matches += 1
                
    retrieval_quality = "HIGH"
    if avg_similarity < 0.85 or keyword_matches < 1:
        retrieval_quality = "LOW"
    
    web_search_triggered = False
    web_domains_used = []
    web_search_ms = 0
    
    if not ranked_chunks or retrieval_quality == "LOW" or max_sim < 0.82 or confidence_score < 0.02:
        from radlearn.retrieval.web_search import perform_web_search
        t_web = time.time()
        
        search_term = " ".join(concepts) if concepts else question
        web_chunks = perform_web_search(search_term)
        
        if web_chunks:
            web_search_triggered = True
            web_domains_used = ["PubMed"]
            
            # Since web chunks are highly relevant fallback, prepend them to ranked_chunks
            # We assign them a synthetic RRF and similarity score to guarantee they appear in the LLM prompt
            # and prevent the LOW_CONFIDENCE_PROMPT from being used.
            for i, wc in enumerate(web_chunks):
                wc["rrf_score"] = 1.0 - (i * 0.01)
                wc["similarity_score"] = 0.99
                
            ranked_chunks = (web_chunks + ranked_chunks)[:top_n]
            
            confidence_score = ranked_chunks[0]["rrf_score"] if ranked_chunks else 0.0
            max_sim = max([c.get("similarity_score", 0.0) for c in ranked_chunks]) if ranked_chunks else 0.0
            
        web_search_ms = int((time.time() - t_web) * 1000)

    if max_sim > 0.84 and len(ranked_chunks) >= 2:
        confidence_band = "🟢 High Confidence"
    elif max_sim >= 0.82:
        confidence_band = "🟡 Medium Confidence"
    else:
        confidence_band = "🔴 Low Confidence"
    
    # 5. Prompt Build
    if not ranked_chunks or (max_sim < 0.82 and not web_search_triggered):
        sys_prompt = LOW_CONFIDENCE_PROMPT
    else:
        sys_prompt = SYSTEM_PROMPT
        if project_id:
            from radlearn.database.projects import get_project
            project = get_project(project_id)
            if project and project.get("system_instructions"):
                # Append the custom instructions to the base system prompt
                sys_prompt = sys_prompt + "\n\nPROJECT CUSTOM INSTRUCTIONS:\n" + project["system_instructions"]
        
    t1 = time.time()
    user_prompt = build_user_prompt(question, ranked_chunks, q_type=q_type, concepts=concepts)
    prompt_build_ms = int((time.time() - t1) * 1000)
    prompt_tokens = len(sys_prompt + user_prompt) // 4
    
    # 6. Gemini Call
    if not ranked_chunks:
        answer = "Information not available in knowledge base."
        citations = []
        llm_gen_ms = 0
    else:
        t2 = time.time()
        try:
            answer = get_llm_client().generate_answer(sys_prompt, user_prompt)
            llm_gen_ms = int((time.time() - t2) * 1000)
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
                return {
                    "status": "quota_exceeded",
                    "message": "Something went wrong, try again later.",
                    "error_details": str(e)
                }
            elif "400" in error_str or "401" in error_str or "api key not valid" in error_str or "api_key_invalid" in error_str or "authentication" in error_str:
                return {
                    "status": "invalid_api_key",
                    "message": "The Google API key is invalid. Please check your .env file.",
                    "error_details": str(e)
                }
            raise e
            
        # 1. Strip out PDF bibliography pollution (any existing [N] brackets)
        answer = re.sub(r'\[[\d\s,\-]+\]', '', answer)
        
        # 2. Normalize grouped RAG citations (e.g., <<2, 4, 5>> -> <<2>> <<4>> <<5>>)
        def _repl(m):
            nums = re.findall(r'\d+', m.group(1))
            return " ".join([f"<<{n}>>" for n in nums])
        answer = re.sub(r'<<([\d\s,]+)>>', _repl, answer)
        
        # 3. Convert unique RAG markers back to standard UI markers (<<N>> -> [N])
        answer = re.sub(r'<<(\d+)>>', r'[\1]', answer)
        
        # 7. Citation Validation
        citations = parse_citations(answer, ranked_chunks)
        
    return {
        "status": "success",
        "answer": answer,
        "citations": citations,
        "confidence_score": confidence_score,
        "confidence_band": confidence_band,
        "retrieved_chunks": ranked_chunks,
        "expanded_query": expanded_query,
        "prompt_sent": {
            "system": sys_prompt,
            "user": user_prompt
        },
        "retrieval_ms": retrieval_ms,
        "prompt_build_ms": prompt_build_ms,
        "llm_gen_ms": llm_gen_ms,
        "total_ms": retrieval_ms + prompt_build_ms + llm_gen_ms + web_search_ms,
        "prompt_tokens": prompt_tokens,
        "web_search_triggered": web_search_triggered,
        "web_domains_used": web_domains_used,
        "web_search_ms": web_search_ms,
        "retrieval_quality": retrieval_quality,
        "avg_similarity": avg_similarity,
        "keyword_matches": keyword_matches
    }
