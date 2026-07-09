"""
tests/test_citation_parser.py
─────────────────────────────
Unit tests for citation parser.
"""
import unittest
from radlearn.chat.citation_parser import parse_citations

class TestCitationParser(unittest.TestCase):
    def test_parse_valid_citations(self):
        answer = "This is a fact [1] and another fact [2]."
        chunks = [
            {"id": "c1", "document_id": "d1", "text": "fact 1", "doc_author": "Smith"},
            {"id": "c2", "document_id": "d2", "text": "fact 2", "doc_author": "Jones"}
        ]
        cites = parse_citations(answer, chunks)
        
        self.assertEqual(len(cites), 2)
        self.assertEqual(cites[0]["citation_number"], 1)
        self.assertEqual(cites[1]["chunk_id"], "c2")
        
    def test_ignore_hallucinated_citations(self):
        answer = "Fact [1]. Fake [3]."
        chunks = [{"id": "c1", "document_id": "d1", "text": "fact 1"}]
        cites = parse_citations(answer, chunks)
        
        self.assertEqual(len(cites), 1) # [3] is ignored
        self.assertEqual(cites[0]["citation_number"], 1)

if __name__ == "__main__":
    unittest.main()
