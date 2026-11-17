"""Chunking strategies for Python, Markdown and generic text files.

    A strategy only proposes cut points (character offsets); the shared
    split_oversized and merge_pieces functions turn them into chunks that tile
    the file and never exceed the maximum size.
"""

import ast
from typing import List, Tuple
from src.models import Chunk, ChunkKind
Span = Tuple[int, int]


def line_starts(text: str) -> List[int]:
    """Return the offset of the first character of every line.

    Args:
        text: Full file content.

    Returns:
        Offsets in increasing order, starting with 0. A trailing newline
        does not open a new line.
    """

    offset = [0]

    for index, char in enumerate(text):
        if char == "\n":
            if index + 1 < len(text):
                offset.append(index+1)
    return offset


def split_oversized(text: str, cuts: List[int], max_size: int) -> List[Span]:
    """Turn cut points into contiguous pieces no wider than max_size.

    Segments between consecutive cuts are kept whole when they fit. A
    segment that is too wide is split at its line starts, and a line that
    is still too wide is cut every max_size characters.

    Args:
        text: Full file content.
        cuts: Candidate cut offsets, in any order; 0 and len(text) are
            always added.
        max_size: Maximum piece width in characters.

    Returns:
        (start, end) pieces that cover the text without gaps.
    """

    limits = [0] + [len(text)]
    for elem in cuts:
        if 0 < elem < len(text):
            limits.append(elem)
    limits = sorted(set(limits))

    result = []
    starts = line_starts(text)
    for start, end in zip(limits, limits[1:]):
        if (end - start <= max_size):
            result.append((start, end))
        else:
            sub_limit = [start]
            for elem in starts:
                if start < elem < end:
                    sub_limit.append(elem)
            sub_limit.append(end)

            for a, b in zip(sub_limit, sub_limit[1:]):
                while (b - a) > max_size:
                    result.append((a, a + max_size))
                    a = a + max_size
                result.append((a, b))

    return result


def merge_pieces(pieces: List[Span], max_size: int) -> List[Span]:
    """Greedily join consecutive pieces while the result fits in max_size.

    Args:
        pieces: Contiguous (start, end) pieces, each no wider than max_size.
        max_size: Maximum chunk width in characters.

    Returns:
        The merged (start, end) spans, still contiguous.
    """
    box = []
    for start, end in pieces:
        if box and end - box[-1][0] <= max_size:
            box[-1] = (box[-1][0], end)
        else:
            box.append((start, end))

    return box


def node_start(node: ast.stmt, starts: List[int]) -> int:
    """Return the offset where a statement begins, decorators included.

    Args:
        node: A statement from an ast tree.
        starts: Line start offsets of the source, from line_starts.

    Returns:
        Offset of the first decorator line if the node has decorators,
        otherwise of its own first line.
    """

    decorators = getattr(node, "decorator_list", [])
    line = node.lineno
    if decorators:
        line = decorators[0].lineno
    return starts[line - 1]


def python_cuts(text: str, max_size: int) -> List[int]:
    """Propose cut points at the top-level statements of a Python file.

    Each top-level statement starts a new segment, from its first
    decorator if it has one. A class wider than max_size is also cut at
    each of its methods.

    Args:
        text: Python source code.
        max_size: Maximum chunk width, used to detect big classes.

    Returns:
        Sorted cut offsets."""

    tree = ast.parse(text)
    starts = line_starts(text)
    cuts = []
    for node in tree.body:
        cuts.append(node_start(node, starts))

        if isinstance(node, ast.ClassDef):
            last_line = node.end_lineno or node.lineno
            if last_line < len(starts):
                class_end = starts[last_line]
            else:
                class_end = len(text)
            if class_end - node_start(node, starts) > max_size:
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        cuts.append(node_start(m, starts))

    return cuts


def markdown_cuts(text: str, max_size: int) -> List[int]:
    """Propose cut points at the headers of a Markdown file.

    A header is a line starting with one to six '#' and a space, outside
    fenced code blocks. A section wider than max_size is also cut at the
    first line of each of its paragraphs.

    Args:
        text: Markdown content.
        max_size: Maximum chunk width, used to detect big sections.

    Returns:
        Sorted cut offsets.
    """
    starts = line_starts(text)
    headers = []
    paragraphs = []
    in_fence = False
    previous_blank = False

    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        line = text[start:end]
        stripped = line.strip()

        if not in_fence:
            rest = line.lstrip("#")
            hashes = len(line) - len(rest)
            if 1 <= hashes <= 6 and rest.startswith(" "):
                headers.append(start)
            elif previous_blank and stripped != "":
                paragraphs.append(start)

        if stripped.startswith("```"):
            in_fence = not in_fence

        previous_blank = stripped == ""

    cuts = list(headers)
    bounds = [0] + headers + [len(text)]
    for a, b in zip(bounds, bounds[1:]):
        if b - a > max_size:
            for p in paragraphs:
                if a < p < b:
                    cuts.append(p)

    return sorted(cuts)


def text_cuts(text: str) -> List[int]:
    """Propose cut points at the paragraphs of a generic text file.

    Args:
        text: File content.

    Returns:
        Sorted offsets of every non-blank line that follows a blank line.
    """
    starts = line_starts(text)
    previous_blank = False
    cuts = []

    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        line = text[start:end]
        stripped = line.strip()

        if previous_blank and stripped != "":
            cuts.append(start)
        previous_blank = stripped == ""

    return cuts


def chunk_file(file_path: str, text: str, kind: ChunkKind,
               max_size: int) -> List[Chunk]:
    """Split one file into chunks with the strategy matching its kind.

    Python files that cannot be parsed fall back to the generic strategy.

    Args:
        file_path: Corpus path stored in every chunk.
        text: Full file content.
        kind: Chunking strategy to use.
        max_size: Maximum chunk width in characters.

    Returns:
        Chunks that cover the whole text in order; empty for empty text.
    """

    if kind == "python":
        try:
            cuts = python_cuts(text, max_size)
        except (SyntaxError, ValueError):
            cuts = text_cuts(text)
    elif kind == "markdown":
        cuts = markdown_cuts(text, max_size)
    else:
        cuts = text_cuts(text)

    merged = split_oversized(text, cuts, max_size)
    spans = merge_pieces(merged, max_size)

    chunks = []
    for start, end in spans:
        chunks.append(Chunk(file_path=file_path,
                            first_character_index=start,
                            last_character_index=end,
                            kind=kind,
                            text=text[start:end]))

    return chunks
