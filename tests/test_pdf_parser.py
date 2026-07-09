"""
tests/test_pdf_parser.py
─────────────────────────
Unit tests for the PDF Parser.
"""
import unittest
from unittest.mock import MagicMock, patch
from radlearn.ingestion.pdf_parser import parse_pdf

class TestPDFParser(unittest.TestCase):
    
    @patch('radlearn.ingestion.pdf_parser.fitz')
    def test_parse_pdf_extracts_blocks(self, mock_fitz):
        # Mock document and page
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        
        mock_page.rect.width = 600
        # Blocks: (x0, y0, x1, y1, text, block_no, block_type)
        # Type 0 is text. Type 1 is image.
        mock_page.get_text.return_value = [
            (50, 100, 200, 200, "Left column text", 0, 0),
            (350, 100, 500, 200, "Right column text", 1, 0),
            (50, 50, 100, 60, "LOGO", 2, 1) # Image block, should be ignored
        ]
        
        pages_data = parse_pdf("dummy.pdf")
        
        self.assertEqual(len(pages_data), 1)
        self.assertIn("Left column text", pages_data[0]["text"])
        self.assertIn("Right column text", pages_data[0]["text"])
        self.assertNotIn("LOGO", pages_data[0]["text"])
        self.assertEqual(pages_data[0]["blocks"], 2)

if __name__ == "__main__":
    unittest.main()
