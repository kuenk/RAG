"""Pydantic data models exchanged between the stages of the RAG pipeline."""

class MinimalSource:
    """A character range inside one file of the corpus.

    Attributes:
        file_path: Path of the file, exactly as in the corpus
            (e.g. ``data/raw/vllm-0.10.1/docs/features/lora.md``).
        first_character_index: Index of the first character (inclusive).
        last_character_index: Index after the last character (exclusive),
            so the covered text is ``content[first:last]``.
    """

    def check_range(self):
        """Validate that the range is non-empty and fits the grader limit.

        Returns:
            The validated source.

        Raises:
            ValueError: If the start is negative, the range is empty or
                inverted, or it is wider than MAX_SOURCE_LENGTH.
        """

class UnansweredQuestion:
    """A question to answer, identified by a unique id."""

class AnsweredQuestion:
    """A question with its reference sources and answer."""

class RagDataset:
    """A dataset of questions, answered or not."""

class MinimalSearchResults:
    """The sources retrieved for one question."""

class MinimalAnswer:
    """The sources retrieved for one question and the generated answer."""

class StudentSearchResults:
    """Search results for a whole dataset, as written by search_dataset."""

class StudentSearchResultsAndAnswer:
    """Search results and answers for a dataset, as written by answer_dataset."""

class Chunk:
    """A piece of a corpus file, stored in the index with its text.

    Attributes:
        kind: Chunking strategy that produced it.
        text: The chunk content, equal to ``content[first:last]``.
    """

    def check_text_length(self):
        """Validate that the text length matches the character range.

        Returns:
            The validated chunk.

        Raises:
            ValueError: If ``len(text)`` differs from the range width.
        """

    def to_source(self):
        """Return the chunk location without its text.

        Returns:
            A MinimalSource with the same file path and range.
        """

class IndexManifest:
    """Configuration and identity of a built index.

    Attributes:
        index_version: Random id regenerated on every build.
        max_chunk_size: Chunk size limit used for the build.
        num_chunks: Number of chunks in the index.
        bm25_k1: BM25 term-frequency saturation parameter.
        bm25_b: BM25 length-normalisation parameter.
    """