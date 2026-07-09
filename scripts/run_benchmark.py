#!/usr/bin/env python3
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_retrieval import run_retrieval

queries = [
    "What does BI-RADS 4 indicate and how should it be managed?",
    "What supplemental screening options are recommended for women with dense breasts?",
    "At what age should routine breast cancer screening begin for average-risk women?",
    "What imaging modalities are recommended for suspected intracranial aneurysm?",
    "What imaging is appropriate for subarachnoid hemorrhage evaluation?",
    "How are vascular malformations evaluated radiologically?",
    "What are the major diagnostic criteria for tuberous sclerosis complex?",
    "What imaging findings are associated with tuberous sclerosis complex?",
    "What is the recommended imaging workup for chronic cough?",
    "When is CT indicated in the evaluation of chronic cough?",
    "What imaging is appropriate for suspected diffuse lung disease?",
    "How should diffuse lung disease be followed over time?",
    "What is the preferred imaging modality for suspected axial spondyloarthritis?",
    "When should imaging be obtained for low back pain?",
    "What imaging findings support a diagnosis of spinal infection?",
    "What is the initial imaging test of choice for suspected spine infection?",
    "How should an incidental adrenal mass be evaluated?",
    "What imaging modalities are recommended for localization of parathyroid adenoma?",
    "What imaging is appropriate in a patient with suspected acute aortic syndrome?",
    "What is the recommended imaging approach for suspected retroperitoneal hemorrhage?",
    "What imaging modality is most appropriate for plexopathy?",
    "How is large vessel vasculitis evaluated with imaging?",
    "What imaging is recommended for rectovaginal fistula?",
    "What is the role of MRI in staging vaginal cancer?",
    "What are MRI findings of multiple sclerosis?",
    "What are LI-RADS criteria for hepatocellular carcinoma?",
    "How is acute pancreatitis staged on CT?",
    "What are Fleischner guidelines for pulmonary nodules?"
]

results = []

for idx, q in enumerate(queries):
    print(f"Running query {idx+1}/{len(queries)}: {q}")
    res = run_retrieval(q)
    
    top_5 = []
    for c in res['ranked_chunks'][:5]:
        top_5.append({
            "doc_title": c.get("doc_title", "Unknown"),
            "page_number": c.get("page_number"),
            "semantic_score": c.get("similarity_score", 0.0),
            "keyword_score": 0.0, # Not easily extracted from RRF output alone without modifying searcher, but RRF is there
            "rrf_score": c.get("rrf_score", 0.0),
            "text": c.get("text", "")
        })
    
    results.append({
        "question": q,
        "top_5": top_5
    })

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done.")
