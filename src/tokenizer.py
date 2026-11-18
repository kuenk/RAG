"""Tokenization shared by indexing and search.

The same tokenize function must turn both the chunks and the queries into
tokens: if it changes, the index has to be rebuilt.
"""

import re
from typing import List

STOPWORDS = frozenset("""
a an the and or but of to in on for with by from at as into about over under
is are was were be been being am do does did done have has had having
can could should would will shall may might must
i you he she it we they me him her us them my your his its our their
this that these those there here what which who whom whose when where why how
if then than so not no yes also just only very
use used using vllm
""".split())

WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def split_camel_case(word: str) -> List[str]:
    """Split a camelCase or PascalCase word into its parts.

    A new part starts at an uppercase letter that follows a lowercase
    letter or a digit, or that ends a run of uppercase letters and is
    followed by a lowercase one ("HTTPServer" -> "HTTP", "Server").

    Args:
        word: A word without underscores.

    Returns:
        The parts in order, with their original case; empty for "".
    """
    parts = []
    current = ""

    for i, char in enumerate(word):
        previous = word[i - 1] if i > 0 else ""
        following = word[i + 1] if i + 1 < len(word) else ""

        starts_word = char.isupper() and (
            previous.islower() or previous.isdigit()
            or (previous.isupper() and following.islower()))

        if starts_word and current:
            parts.append(current)
            current = ""

        current += char

    if current:
        parts.append(current)

    return parts

def split_identifier(word: str) -> List[str]:
    """Split an identifier at underscores and camelCase boundaries.

    Args:
        word: An identifier such as "max_model_len" or "FusedMoEMethod".

    Returns:
        The lowercase parts in order, without empty parts.
    """
    raise NotImplementedError


def normalize_plural(word: str) -> str:
    """Turn a simple English plural into its singular.

    Args:
        word: A lowercase token.

    Returns:
        The word without its final "s" if it is longer than three
        characters and does not end in "ss"; otherwise the word unchanged.
    """
    raise NotImplementedError


def tokenize(text: str) -> List[str]:
    """Turn text into the tokens used by BM25.

    Every word is kept whole in lowercase and, when it is a compound
    identifier, also split into its parts. Stopwords are dropped and
    plurals are normalised.

    Args:
        text: A chunk of the corpus, a file path or a user question.

    Returns:
        The tokens in order of appearance, repetitions included.
    """
    raise NotImplementedError


def document_tokens(file_path: str, text: str) -> List[str]:
    """Return the tokens indexed for a chunk: its text plus its file path.

    The path names the topic of the file, so its words help to match
    questions whose chunk text does not repeat them.

    Args:
        file_path: Corpus path of the chunk.
        text: Text of the chunk.

    Returns:
        The tokens of the text followed by the tokens of the path.
    """
    raise NotImplementedError
