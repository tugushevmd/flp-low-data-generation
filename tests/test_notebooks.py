from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


class NotebookTests(unittest.TestCase):
    def test_notebooks_are_valid_json(self):
        for path in NOTEBOOKS.glob("*.ipynb"):
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("cells", notebook)

    def test_notebooks_do_not_use_local_windows_paths(self):
        for path in NOTEBOOKS.glob("*.ipynb"):
            with self.subTest(notebook=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("C:\\\\Users", text)
                self.assertNotIn("Downloads", text)

    def test_notebooks_do_not_replace_platform_pytorch(self):
        for path in NOTEBOOKS.glob("*.ipynb"):
            with self.subTest(notebook=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("pip install -q torch==", text)


if __name__ == "__main__":
    unittest.main()
