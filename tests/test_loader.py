"""Tests for document loading."""


import pytest

from src.ingestion.loader import load_directory, load_markdown


class TestLoadMarkdown:
    """Tests for load_markdown()."""

    def test_load_valid_markdown(self, tmp_path):
        """Loading a valid markdown file returns text and metadata."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nThis is a test.", encoding="utf-8")

        result = load_markdown(str(md_file))

        assert result["text"] == "# Hello\n\nThis is a test."
        assert result["metadata"]["filename"] == "test.md"
        assert "path" in result["metadata"]

    def test_load_missing_file_raises(self):
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_markdown("/nonexistent/path/file.md")

    def test_load_empty_file(self, tmp_path):
        """Loading an empty file returns empty text."""
        md_file = tmp_path / "empty.md"
        md_file.write_text("", encoding="utf-8")

        result = load_markdown(str(md_file))
        assert result["text"] == ""


class TestLoadDirectory:
    """Tests for load_directory()."""

    def test_load_directory_with_matching_files(self, tmp_path):
        """Loading a directory returns all matching files."""
        (tmp_path / "a.md").write_text("Doc A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Doc B", encoding="utf-8")
        (tmp_path / "c.json").write_text("{}", encoding="utf-8")  # should be skipped

        results = load_directory(str(tmp_path))

        assert len(results) == 2
        texts = [r["text"] for r in results]
        assert "Doc A" in texts
        assert "Doc B" in texts

    def test_load_directory_custom_extensions(self, tmp_path):
        """Custom extensions filter works."""
        (tmp_path / "code.py").write_text("print('hi')", encoding="utf-8")
        (tmp_path / "notes.md").write_text("Notes", encoding="utf-8")

        results = load_directory(str(tmp_path), extensions=[".py"])

        assert len(results) == 1
        assert results[0]["metadata"]["filename"] == "code.py"

    def test_load_nonexistent_directory_raises(self):
        """Loading a nonexistent directory raises NotADirectoryError."""
        with pytest.raises(NotADirectoryError):
            load_directory("/nonexistent/dir")

    def test_load_empty_directory(self, tmp_path):
        """Loading an empty directory returns an empty list."""
        results = load_directory(str(tmp_path))
        assert results == []
