"""Command-line interface of the RAG system, exposed through Python Fire."""

class RagCLI:
    """Commands to index the vLLM corpus, search it and answer questions."""

    def index(...):
        """Chunk the corpus and persist the lexical index.

        Args:
            max_chunk_size: Maximum chunk length in characters (1-2000).
            corpus_dir: Directory containing the raw sources to index.
            index_dir: Directory where the index files are written.
        """

    def search(...):
        """Print the top-k sources for a single query.

        Args:
            query: Natural-language question.
            k: Number of sources to return.
            index_dir: Directory containing the index files.
        """

    def search_dataset(...):
        """Search every question of a dataset and save the results.

        Args:
            dataset_path: Path to a RagDataset JSON file.
            save_directory: Directory where the StudentSearchResults JSON
                is written, under the dataset's file name.
            k: Number of sources to return per question.
            index_dir: Directory containing the index files.
        """

    def answer(...):
        """Answer a single query from the retrieved context.

        Args:
            query: Natural-language question.
            k: Number of sources given to the model as context.
            index_dir: Directory containing the index files.
        """

    def answer_dataset(...):
        """Generate answers for previously saved search results.

        Args:
            student_search_results_path: Path to a StudentSearchResults JSON.
            save_directory: Directory where the StudentSearchResultsAndAnswer
                JSON is written, under the input file name.
            index_dir: Directory containing the index files.
        """

    def evaluate(...):
        """Report recall@k of search results against a ground-truth dataset.

        Args:
            student_search_results_path: Path to a StudentSearchResults JSON.
            dataset_path: Path to an AnsweredQuestions RagDataset JSON.
        """