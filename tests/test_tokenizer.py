"""Checkpoint tests for phase 3: tokenization."""

from typing import List

import pytest

from src.tokenizer import (
    STOPWORDS,
    document_tokens,
    normalize_plural,
    split_camel_case,
    split_identifier,
    tokenize,
)


@pytest.mark.parametrize(
    "word, expected",
    [("FusedMoEMethod", ["Fused", "Mo", "E", "Method"]),
     ("HTTPServer", ["HTTP", "Server"]),
     ("Qwen3ForCausalLM", ["Qwen3", "For", "Causal", "LM"]),
     ("lowercase", ["lowercase"]),
     ("", [])],
)
def test_split_camel_case(word: str, expected: List[str]) -> None:
    """Parts start at lower-to-upper and acronym-to-word boundaries."""
    assert split_camel_case(word) == expected


@pytest.mark.parametrize(
    "word, expected",
    [("max_model_len", ["max", "model", "len"]),
     ("load_LoRAConfig", ["load", "lo", "ra", "config"]),
     ("__init__", ["init"]),
     ("fp8", ["fp8"])],
)
def test_split_identifier(word: str, expected: List[str]) -> None:
    """Identifiers split at underscores and camelCase, in lowercase."""
    assert split_identifier(word) == expected


@pytest.mark.parametrize(
    "word, expected",
    [("adapters", "adapter"), ("layers", "layer"), ("class", "class"),
     ("process", "process"), ("bus", "bus"), ("is", "is"),
     ("model", "model")],
)
def test_normalize_plural(word: str, expected: str) -> None:
    """Only a final 's' of words longer than 3, not in 'ss', is removed."""
    assert normalize_plural(word) == expected


def test_tokenize_question() -> None:
    """Whole words, identifier parts, no stopwords, singular forms."""
    text = "How do I load LoRA adapters with max_model_len?"
    assert tokenize(text) == ["load", "lora", "lo", "ra", "adapter",
                              "max_model_len", "max", "model", "len"]


def test_tokenize_drops_stopwords_in_any_case() -> None:
    """Stopwords are removed after lowercasing, including 'vllm'."""
    assert tokenize("What does VLLM use?") == []
    assert "vllm" in STOPWORDS


def test_tokenize_keeps_repetitions() -> None:
    """Repeated words are kept: BM25 needs their frequency."""
    assert tokenize("cache cache CACHE") == ["cache", "cache", "cache"]


def test_tokenize_empty_and_symbols() -> None:
    """Text without letters or digits gives no tokens."""
    assert tokenize("") == []
    assert tokenize("-- ?? !! ()") == []


def test_document_tokens_include_path() -> None:
    """Path words are appended after the text tokens."""
    path = "data/raw/vllm-0.10.1/vllm/model_executor/fused_batched_moe.py"
    tokens = document_tokens(path, "return x")
    assert tokens[:2] == ["return", "x"]
    assert tokens[2:] == tokenize(path)
    assert {"fused", "batched", "moe", "executor"} <= set(tokens)
