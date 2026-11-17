"""Checkpoint tests for phase 0: data models against the public datasets."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models import (
    MAX_SOURCE_LENGTH,
    AnsweredQuestion,
    Chunk,
    IndexManifest,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)

DATASETS = Path("data/datasets")
PREFIX = "data/raw/vllm-0.10.1/"


def load_dataset(path: Path) -> RagDataset:
    """Read and validate a dataset file."""
    with open(path, encoding="utf-8") as file:
        return RagDataset.model_validate_json(file.read())


@pytest.mark.parametrize(
    "name, expected", [("dataset_docs_public.json", 100),
                       ("dataset_code_public.json", 99)]
)
def test_answered_datasets_parse(name: str, expected: int) -> None:
    """Answered datasets load as AnsweredQuestion with exact paths."""
    dataset = load_dataset(DATASETS / "AnsweredQuestions" / name)
    assert len(dataset.rag_questions) == expected
    for item in dataset.rag_questions:
        assert isinstance(item, AnsweredQuestion)
        for source in item.sources:
            assert source.file_path.startswith(PREFIX)
            assert Path(source.file_path).is_file()


@pytest.mark.parametrize(
    "name", ["dataset_docs_public.json", "dataset_code_public.json"]
)
def test_unanswered_datasets_parse(name: str) -> None:
    """Unanswered datasets load as plain UnansweredQuestion."""
    dataset = load_dataset(DATASETS / "UnansweredQuestions" / name)
    for item in dataset.rag_questions:
        assert isinstance(item, UnansweredQuestion)
        assert not isinstance(item, AnsweredQuestion)


def test_reference_ranges_are_python_slices() -> None:
    """A reference range is text[first:last] counted in characters."""
    path = DATASETS / "AnsweredQuestions" / "dataset_docs_public.json"
    item = load_dataset(path).rag_questions[0]
    assert isinstance(item, AnsweredQuestion)
    source = item.sources[0]
    with open(source.file_path, encoding="utf-8") as file:
        text = file.read()
    span = text[source.first_character_index:source.last_character_index]
    assert span.startswith("### Using API Endpoints")
    assert span.endswith("```")


def test_question_id_is_generated() -> None:
    """A question without id receives a fresh uuid."""
    first = UnansweredQuestion(question="a")
    second = UnansweredQuestion(question="a")
    assert first.question_id != second.question_id


def test_minimal_source_accepts_max_width() -> None:
    """A range of exactly MAX_SOURCE_LENGTH characters is valid."""
    source = MinimalSource(file_path="f", first_character_index=10,
                           last_character_index=10 + MAX_SOURCE_LENGTH)
    assert source.last_character_index == 2010


@pytest.mark.parametrize(
    "first, last", [(-1, 5), (5, 5), (8, 3), (0, MAX_SOURCE_LENGTH + 1)]
)
def test_minimal_source_rejects_bad_ranges(first: int, last: int) -> None:
    """Negative, empty, inverted and over-long ranges are rejected."""
    with pytest.raises(ValidationError):
        MinimalSource(file_path="f", first_character_index=first,
                      last_character_index=last)


def test_chunk_text_must_match_range() -> None:
    """A chunk whose text length differs from its range is rejected."""
    with pytest.raises(ValidationError):
        Chunk(file_path="f", first_character_index=0,
              last_character_index=5, kind="python", text="abc")


def test_chunk_to_source() -> None:
    """to_source keeps only the three fields the grader reads."""
    chunk = Chunk(file_path="f", first_character_index=2,
                  last_character_index=5, kind="markdown", text="abc")
    source = chunk.to_source()
    assert type(source) is MinimalSource
    assert source.model_dump() == {"file_path": "f",
                                   "first_character_index": 2,
                                   "last_character_index": 5}


@pytest.mark.parametrize("size", [0, MAX_SOURCE_LENGTH + 1])
def test_manifest_rejects_bad_chunk_size(size: int) -> None:
    """The manifest only accepts chunk sizes in 1..MAX_SOURCE_LENGTH."""
    with pytest.raises(ValidationError):
        IndexManifest(index_version="v", max_chunk_size=size,
                      num_chunks=0, bm25_k1=1.5, bm25_b=0.75)


def test_search_results_json_round_trip() -> None:
    """Search results survive a JSON round trip unchanged."""
    source = MinimalSource(file_path=PREFIX + "README.md",
                           first_character_index=0, last_character_index=9)
    results = StudentSearchResults(
        search_results=[MinimalSearchResults(
            question_id="q1", question="?", retrieved_sources=[source])],
        k=1,
    )
    text = results.model_dump_json()
    assert StudentSearchResults.model_validate_json(text) == results
