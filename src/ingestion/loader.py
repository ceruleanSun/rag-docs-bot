"""Document loading utilities for the RAG pipeline."""

import os
from pathlib import Path


def load_markdown(file_path: str) -> dict:
    """Load a single markdown file and return its text with metadata.

    Args:
        file_path: Path to the markdown file.

    Returns:
        Dictionary with 'text' (file contents) and 'metadata' (filename, path).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    return {
        "text": text,
        "metadata": {
            "filename": path.name,
            "path": str(path.resolve()),
        },
    }


def load_directory(
    dir_path: str, extensions: list[str] | None = None
) -> list[dict]:
    """Load all matching files from a directory.

    Args:
        dir_path: Path to the directory to scan.
        extensions: List of file extensions to include (e.g. ['.md', '.py']).
            Defaults to ['.md', '.py', '.txt'].

    Returns:
        List of document dicts, each with 'text' and 'metadata'.

    Raises:
        NotADirectoryError: If dir_path is not a directory.
    """
    if extensions is None:
        extensions = [".md", ".py", ".txt"]

    dir_path_obj = Path(dir_path)
    if not dir_path_obj.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    documents = []
    for root, _dirs, files in os.walk(dir_path_obj):
        for filename in sorted(files):
            if any(filename.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, filename)
                try:
                    doc = load_markdown(full_path)
                    documents.append(doc)
                except (UnicodeDecodeError, OSError) as e:
                    print(f"Warning: skipping {full_path}: {e}")

    return documents
