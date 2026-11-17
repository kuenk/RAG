from pydantic import BaseModel, Field, model_validator
from typing import List, Literal
import uuid

MAX_SOURCE_LENGTH = 2000
ChunkKind = Literal["python", "markdown", "text"]


class MinimalSource(BaseModel):
    """A character range inside one file of the corpus.

    Attributes:
        file_path: Path of the file, exactly as in the corpus
            (e.g. ``data/raw/vllm-0.10.1/docs/features/lora.md``).
        first_character_index: Index of the first character (inclusive).
        last_character_index: Index after the last character (exclusive),
            so the covered text is ``content[first:last]``.
    """
    file_path: str
    first_character_index: int
    last_character_index: int

    @model_validator(mode="after")
    def check_range(self) -> "MinimalSource":
        """Validate that the range is non-empty and fits the grader limit.

        Returns:
            The validated source.

        Raises:
            ValueError: If the start is negative, the range is empty or
                inverted, or it is wider than MAX_SOURCE_LENGTH.
        """
        if self.first_character_index < 0:
            raise ValueError("first_character_index must be >= 0")
        if self.last_character_index <= self.first_character_index:
            raise ValueError("last_character_index must be more than "
                             "first_character_index")
        if (self.last_character_index - self.first_character_index >
                MAX_SOURCE_LENGTH):
            raise ValueError("first_character_index - last_character_index "
                             "must be >= 0")
        return self


class UnansweredQuestion(BaseModel):
    """A question to answer, identified by a unique id."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, answered or not."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The sources retrieved for one question and the generated answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset, as written by search_dataset."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and answers for a dataset, as written by
    answer_dataset."""
    search_results: List[MinimalAnswer]
    k: int


class Chunk(MinimalSource):
    """A piece of a corpus file, stored in the index with its text.

    Attributes:
        kind: Chunking strategy that produced it.
        text: The chunk content, equal to ``content[first:last]``.
    """
    kind: ChunkKind
    text: str

    @model_validator(mode="after")
    def check_text_length(self) -> "Chunk":
        """Validate that the text length matches the character range.

        Returns:
            The validated chunk.

        Raises:
            ValueError: If ``len(text)`` differs from the range width.
        """

        if (len(self.text) != self.last_character_index -
                self.first_character_index):
            raise ValueError("The length of both params aren't equal")
        return self

    def to_source(self) -> MinimalSource:
        """Return the chunk location without its text.

        Returns:
            A MinimalSource with the same file path and range.
        """
        return MinimalSource(file_path=self.file_path,
                             first_character_index=self.first_character_index,
                             last_character_index=self.last_character_index)


class IndexManifest(BaseModel):
    """Configuration and identity of a built index.

    Attributes:
        index_version: Random id regenerated on every build.
        max_chunk_size: Chunk size limit used for the build.
        num_chunks: Number of chunks in the index.
        bm25_k1: BM25 term-frequency saturation parameter.
        bm25_b: BM25 length-normalisation parameter.
    """
    index_version: str
    max_chunk_size: int = Field(gt=0, le=MAX_SOURCE_LENGTH)
    num_chunks: int = Field(gt=0)
    bm25_k1: float = Field(gt=0)
    bm25_b: float = Field(ge=0, le=1)
