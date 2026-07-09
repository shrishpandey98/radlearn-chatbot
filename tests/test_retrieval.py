"""
tests/test_retrieval.py
────────────────────────
Unit tests for preprocessor and searcher.
"""
import unittest
from radlearn.retrieval.preprocessor import preprocess_query

class TestPreprocessor(unittest.TestCase):
    def test_abbreviation_expansion(self):
        res = preprocess_query("What is MS?")
        self.assertIn("multiple sclerosis", res["expanded_query"].lower())
        
    def test_specialty_detection(self):
        res = preprocess_query("chest x-ray findings")
        self.assertEqual(res["specialty"], "chest")

if __name__ == "__main__":
    unittest.main()
