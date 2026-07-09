"""
tests/test_ranker.py
─────────────────────
Unit tests for RRF ranker.
"""
import unittest
from radlearn.retrieval.ranker import reciprocal_rank_fusion

class TestRanker(unittest.TestCase):
    def test_rrf_scoring(self):
        sem = [{"id": "chunk1", "text": "A"}, {"id": "chunk2", "text": "B"}]
        kwd = [{"id": "chunk2", "text": "B"}, {"id": "chunk3", "text": "C"}]
        
        ranked = reciprocal_rank_fusion(sem, kwd, k=60, top_n=10)
        
        # chunk2 should be ranked first because it appears in both
        self.assertEqual(ranked[0]["id"], "chunk2")
        self.assertGreater(ranked[0]["rrf_score"], ranked[1]["rrf_score"])
        
if __name__ == "__main__":
    unittest.main()
