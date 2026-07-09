"""
tests/test_docx_parser.py
──────────────────────────
Unit tests for DOCX Parser.
"""
import unittest
from unittest.mock import MagicMock, patch
from radlearn.ingestion.docx_parser import parse_docx

class TestDOCXParser(unittest.TestCase):

    @patch('radlearn.ingestion.docx_parser.docx.Document')
    def test_parse_docx_sections(self, mock_document_class):
        mock_doc = MagicMock()
        mock_document_class.return_value = mock_doc
        
        p1 = MagicMock()
        p1.text = "Introduction"
        p1.style.name = "Heading 1"
        
        p2 = MagicMock()
        p2.text = "This is a paragraph."
        p2.style.name = "Normal"
        
        mock_doc.paragraphs = [p1, p2]
        
        sections = parse_docx("dummy.docx")
        
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["heading"], "Introduction")
        self.assertEqual(sections[0]["text"], "This is a paragraph.")

if __name__ == "__main__":
    unittest.main()
