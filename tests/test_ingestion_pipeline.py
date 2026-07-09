"""
tests/test_ingestion_pipeline.py
─────────────────────────────────
Unit tests for the ingestion pipeline orchestration.
"""
import unittest
from unittest.mock import patch
from radlearn.ingestion.pipeline import ingest_file

class TestIngestionPipeline(unittest.TestCase):

    @patch('radlearn.ingestion.pipeline.check_duplicate_hash')
    def test_skip_duplicate(self, mock_check):
        mock_check.return_value = "existing_doc_id"
        
        res = ingest_file(b"dummy content", "test.pdf", {"title": "Test"})
        
        self.assertEqual(res["status"], "skipped")
        self.assertEqual(res["doc_id"], "existing_doc_id")

if __name__ == "__main__":
    unittest.main()
