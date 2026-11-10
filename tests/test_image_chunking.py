"""Tests for tall-image chunking (no network or API key needed)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from mistral_ocr_pdf import chunk_tall_image, main


class TestChunkTallImage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mistral-ocr-pdf-test-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def make_image(self, name: str, width: int, height: int) -> Path:
        path = self.tmp / name
        Image.new("RGB", (width, height), "white").save(path)
        return path

    def test_nonpositive_max_ratio_raises_instead_of_hanging(self):
        # max_ratio <= 0 makes no forward progress when searching cut
        # points (chunk_height <= 0), hanging forever. It must be rejected.
        img = self.make_image("tall.png", 20, 200)
        for bad in (0, -1.5):
            with self.subTest(max_ratio=bad):
                with self.assertRaises(ValueError):
                    chunk_tall_image(img, max_ratio=bad)

    def test_short_image_needs_no_chunking(self):
        img = self.make_image("short.png", 100, 100)
        self.assertEqual(chunk_tall_image(img), [img])

    def test_tall_image_is_chunked(self):
        img = self.make_image("tall.png", 20, 200)
        chunks = chunk_tall_image(img, max_ratio=3.0)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.is_file())

    def test_cli_rejects_nonpositive_max_ratio(self):
        # Reached before any file/API work, as a clean argparse error.
        argv = ["mistral-ocr-pdf", "whatever.png", "--max-ratio", "0"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
