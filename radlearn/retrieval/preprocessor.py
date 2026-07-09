"""
radlearn/retrieval/preprocessor.py
───────────────────────────────────
Query preprocessing for retrieval.
Expands medical abbreviations and normalizes queries.
"""
import re

# Common radiology abbreviations
ABBREVIATIONS = {
    "ms": "multiple sclerosis",
    "dwi": "diffusion weighted imaging",
    "adc": "apparent diffusion coefficient",
    "flair": "fluid attenuated inversion recovery",
    "mri": "magnetic resonance imaging",
    "ct": "computed tomography",
    "pet": "positron emission tomography",
    "us": "ultrasound",
    "xr": "x-ray",
    "cns": "central nervous system",
    "hcc": "hepatocellular carcinoma",
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis"
}

SPECIALTIES = [
    "neuro", "breast", "chest", "msk", "abdominal", 
    "cardiothoracic", "pediatric", "interventional", "nuclear", "general"
]

def preprocess_query(query: str) -> dict:
    """
    Cleans, expands, and extracts specialty from query.
    Returns dict with original_query, expanded_query, and specialty.
    """
    clean_query = query.strip()
    words = re.findall(r'\b\w+\b', clean_query.lower())
    
    expanded_words = []
    for word in words:
        if word in ABBREVIATIONS:
            expanded_words.append(ABBREVIATIONS[word])
        else:
            expanded_words.append(word)
            
    expanded_query = clean_query
    if clean_query.lower() != " ".join(expanded_words):
         # simple replacement (case insensitive, word boundary)
         for abbr, expansion in ABBREVIATIONS.items():
             expanded_query = re.sub(rf'\b{abbr}\b', expansion, expanded_query, flags=re.IGNORECASE)
             
    # Very naive specialty detection
    detected_specialty = "general"
    for spec in SPECIALTIES:
        if spec in expanded_query.lower():
            detected_specialty = spec
            break
            
    return {
        "original_query": query,
        "expanded_query": expanded_query,
        "specialty": detected_specialty
    }
