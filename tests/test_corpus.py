"""Checkpoint tests for phase 2: corpus discovery and reading."""

from pathlib import Path

import pytest

from src.corpus import kind_for_path, list_corpus_files, read_text_file

CORPUS = "data/raw"
PREFIX = "data/raw/vllm-0.10.1/"


@pytest.mark.parametrize(
    "path, expected",
    [("a/b/model.py", "python"), ("docs/lora.md", "markdown"),
     ("CMakeLists.txt", "text"), ("config.YAML", "text"),
     ("kernel.cu", "text"), ("data.json", None), ("logo.png", None),
     ("LICENSE", None)],
)
def test_kind_for_path(path: str, expected: str) -> None:
    """Each extension maps to its chunking strategy, or to None."""
    assert kind_for_path(path) == expected


def test_list_corpus_files_content() -> None:
    """The listing has the expected files, exact paths and a stable order."""
    files = list_corpus_files(CORPUS)
    assert PREFIX + "docs/features/lora.md" in files
    assert PREFIX + "CMakeLists.txt" in files
    assert PREFIX + "vllm/__init__.py" in files
    assert all(path.startswith(PREFIX) for path in files)
    assert all(kind_for_path(path) is not None for path in files)
    assert files == sorted(files)


@pytest.mark.parametrize("spelling", ["./data/raw", "data/raw/"])
def test_list_corpus_files_normalises_paths(spelling: str) -> None:
    """Other spellings of the corpus directory give the same paths."""
    assert list_corpus_files(spelling) == list_corpus_files(CORPUS)


def test_list_corpus_files_missing_directory() -> None:
    """A missing corpus directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        list_corpus_files("data/does_not_exist")


def test_read_text_file(tmp_path: Path) -> None:
    """UTF-8 files are read as text; undecodable files give None."""
    good = tmp_path / "good.md"
    good.write_text("# Título ✓\n", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00broken")
    assert read_text_file(str(good)) == "# Título ✓\n"
    assert read_text_file(str(bad)) is None