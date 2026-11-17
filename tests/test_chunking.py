"""Checkpoint tests for phase 2: chunking strategies and packing."""

from typing import List, Tuple

import pytest

from src.chunking import (
    chunk_file,
    line_starts,
    markdown_cuts,
    merge_pieces,
    python_cuts,
    split_oversized,
    text_cuts,
)
from src.corpus import kind_for_path, list_corpus_files, read_text_file
from src.models import AnsweredQuestion, Chunk, RagDataset

PYTHON_SOURCE = (
    '"""Module docstring."""\n'           # 0
    "import os\n"                         # 24
    "\n"                                  # 34
    "\n"                                  # 35
    "@decorator\n"                        # 36
    "def first():\n"                      # 47
    "    return 1\n"                      # 60
    "\n"                                  # 73
    "\n"                                  # 74
    "class Big:\n"                        # 75
    "    def a(self):\n"                  # 86
    "        return 'aaaaaaaaaaaaaaaa'\n"  # 103
    "\n"                                  # 137
    "    @property\n"                     # 138
    "    def b(self):\n"                  # 152
    "        return 'bbbbbbbbbbbbbbbb'\n"  # 169
)

MARKDOWN_SOURCE = (
    "# Title\n"                 # 0
    "\n"                        # 8
    "Intro text.\n"             # 9
    "\n"                        # 21
    "```bash\n"                 # 22
    "# not a header\n"          # 30
    "pip install vllm\n"        # 45
    "```\n"                     # 62
    "\n"                        # 66
    "## Usage\n"                # 67
    "\n"                        # 76
    "First paragraph.\n"        # 77
    "\n"                        # 94
    "Second paragraph.\n"       # 95
)


def check_tiling(chunks: List[Chunk], text: str, max_size: int) -> None:
    """Assert chunks cover text contiguously, each within max_size."""
    position = 0
    for chunk in chunks:
        assert chunk.first_character_index == position
        width = chunk.last_character_index - chunk.first_character_index
        assert 0 < width <= max_size
        assert chunk.text == text[chunk.first_character_index:
                                  chunk.last_character_index]
        position = chunk.last_character_index
    assert position == len(text)


def test_line_starts() -> None:
    """Offsets of every line start; no entry past the last character."""
    assert line_starts("ab\ncd\n") == [0, 3]
    assert line_starts("ab\ncd") == [0, 3]
    assert line_starts("") == [0]


def test_split_oversized_keeps_small_segments() -> None:
    """Segments that fit stay whole, and 0 and len(text) are implicit."""
    assert split_oversized("aaaa\nbbbb\n", [5], 100) == [(0, 5), (5, 10)]


def test_split_oversized_falls_back_to_lines_then_hard_cuts() -> None:
    """A segment too big is cut at lines; a line too big, every max_size."""
    text = "aaaa\nbbbbbbbbbb\n"
    assert split_oversized(text, [], 6) == [(0, 5), (5, 11), (11, 16)]


def test_merge_pieces() -> None:
    """Consecutive pieces are joined while the result fits."""
    pieces: List[Tuple[int, int]] = [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert merge_pieces(pieces, 6) == [(0, 6), (6, 10)]
    assert merge_pieces(pieces, 100) == [(0, 10)]
    assert merge_pieces([], 6) == []


def test_python_cuts_small_class() -> None:
    """Top-level nodes are cut points, starting at their first decorator."""
    assert python_cuts(PYTHON_SOURCE, 2000) == [0, 24, 36, 75]


def test_python_cuts_big_class_adds_methods() -> None:
    """A class larger than max_size is also cut at each of its methods."""
    assert python_cuts(PYTHON_SOURCE, 100) == [0, 24, 36, 75, 86, 138]


def test_markdown_cuts_ignore_fenced_hashes() -> None:
    """Headers are cuts; a '#' line inside a code fence is not."""
    assert markdown_cuts(MARKDOWN_SOURCE, 2000) == [0, 67]


def test_markdown_cuts_split_big_sections_at_paragraphs() -> None:
    """A section larger than max_size is also cut at its paragraphs."""
    assert markdown_cuts(MARKDOWN_SOURCE, 30) == [0, 9, 22, 67, 77, 95]


def test_text_cuts() -> None:
    """Generic text is cut at the first line after each blank line."""
    assert text_cuts("a\nb\n\nc\n\n\nd\n") == [5, 9]


@pytest.mark.parametrize("max_size", [2000, 100, 40, 7])
def test_chunk_file_python(max_size: int) -> None:
    """Python chunks tile the file and respect max_size."""
    chunks = chunk_file("f.py", PYTHON_SOURCE, "python", max_size)
    check_tiling(chunks, PYTHON_SOURCE, max_size)
    assert all(chunk.kind == "python" for chunk in chunks)


def test_chunk_file_python_syntax_error_falls_back() -> None:
    """Unparsable Python is still chunked, with the generic strategy."""
    text = "def broken(:\n    pass\n\nx = 1\n"
    chunks = chunk_file("f.py", text, "python", 2000)
    check_tiling(chunks, text, 2000)


@pytest.mark.parametrize("max_size", [2000, 50, 10])
def test_chunk_file_markdown(max_size: int) -> None:
    """Markdown chunks tile the file and respect max_size."""
    chunks = chunk_file("f.md", MARKDOWN_SOURCE, "markdown", max_size)
    check_tiling(chunks, MARKDOWN_SOURCE, max_size)


def test_chunk_file_empty_text() -> None:
    """An empty file produces no chunks."""
    assert chunk_file("f.md", "", "markdown", 2000) == []


@pytest.mark.parametrize("max_size", [2000, 500])
def test_whole_corpus(max_size: int) -> None:
    """Every corpus file tiles correctly at realistic chunk sizes."""
    for path in list_corpus_files("data/raw"):
        text = read_text_file(path)
        kind = kind_for_path(path)
        assert text is not None and kind is not None
        check_tiling(chunk_file(path, text, kind, max_size), text, max_size)


def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Intersection over union of two character ranges."""
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union


@pytest.mark.parametrize(
    "name, minimum", [("dataset_docs_public.json", 95),
                      ("dataset_code_public.json", 95)]
)
def test_references_are_reachable(name: str, minimum: int) -> None:
    """Almost every reference overlaps some chunk with IoU >= 0.05."""
    path = "data/datasets/AnsweredQuestions/" + name
    with open(path, encoding="utf-8") as file:
        dataset = RagDataset.model_validate_json(file.read())
    reachable = 0
    for item in dataset.rag_questions:
        assert isinstance(item, AnsweredQuestion)
        source = item.sources[0]
        text = read_text_file(source.file_path)
        kind = kind_for_path(source.file_path)
        assert text is not None and kind is not None
        reference = (source.first_character_index,
                     source.last_character_index)
        for chunk in chunk_file(source.file_path, text, kind, 2000):
            span = (chunk.first_character_index, chunk.last_character_index)
            if iou(reference, span) >= 0.05:
                reachable += 1
                break
    assert reachable >= minimum