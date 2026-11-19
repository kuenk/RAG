"""BM25 ranking over a sparse term-frequency matrix.

The index stores, for every chunk, how many times each token appears in it.
Everything else BM25 needs (document lengths, their average and the IDF of
each token) is derived from that matrix when the index is created.
"""

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class BM25Index:
    """Lexical index that ranks documents for a query with BM25.

    Attributes:
        vocabulary: Token -> column number in the matrix.
        term_frequencies: Sparse matrix, one row per document and one
            column per token, holding how many times the token appears.
        k1: Term-frequency saturation parameter.
        b: Length-normalisation parameter, between 0 and 1.
        num_documents: Number of rows of the matrix.
        document_lengths: Number of tokens of each document.
        average_length: Mean of document_lengths (0 for an empty index).
        idf: Inverse document frequency of each token, by column.
    """

    def __init__(self, vocabulary: Dict[str, int],
                 term_frequencies: sparse.csc_matrix,
                 k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        """Create an index and derive its statistics from the matrix.

        Args:
            vocabulary: Token -> column number in term_frequencies.
            term_frequencies: Documents x tokens count matrix, CSC format.
            k1: Term-frequency saturation parameter.
            b: Length-normalisation parameter.
        """
        self.vocabulary = vocabulary
        self.term_frequencies = term_frequencies
        self.k1 = k1
        self.b = b
        self.num_documents = term_frequencies.shape[0]
        self.document_lengths = np.asarray(term_frequencies.sum(axis=1),
                                dtype=np.float64).ravel()
        if self.num_documents > 0:
            self.average_length = float(self.document_lengths.mean())
        else:
            self.average_length = 0.0
        document_frequencies = term_frequencies.getnnz(axis=0)
        self.idf = np.log(1.0 + (self.num_documents - document_frequencies + 0.5)
                            /(document_frequencies + 0.5))

    @classmethod
    def build(cls, documents: List[List[str]], k1: float = DEFAULT_K1,
              b: float = DEFAULT_B) -> "BM25Index":
        """Build an index from the tokens of every document.

        Args:
            documents: One list of tokens per document, in document order.
            k1: Term-frequency saturation parameter.
            b: Length-normalisation parameter.

        Returns:
            The index; document i is row i of its matrix.
        """
        vocab = {}
        rows = []
        columns = []
        counts = []

        for row, tokens in enumerate(documents):
            for token, count in Counter(tokens).items():
                if token not in vocab:
                    vocab[token] = len(vocab)
                rows.append(row)
                columns.append(vocab[token])
                counts.append(count)

        matrix = sparse.csc_matrix((counts, (rows, columns)),
                                shape=(len(documents), len(vocab)),
                                dtype=np.float64)

        return cls(vocab, matrix, k1, b)

    def score(self, query_tokens: List[str]) -> NDArray[np.float64]:
        """Compute the BM25 score of every document for a query.

        Each distinct query token counts once; tokens that are not in the
        vocabulary are ignored.

        Args:
            query_tokens: Tokens of the query, from tokenize.

        Returns:
            One score per document; 0 where no query token appears.
        """
        scores = np.zeros(self.num_documents, dtype=np.float64)
        for token in sorted(set(query_tokens)):
            j = self.vocabulary.get(token)
            if j is None:
                continue
            column = self.term_frequencies[:, j]
            rows = column.indices
            tf = column.data

            lengths = self.document_lengths[rows]
            denominator = tf + self.k1 * (
                1 - self.b + self.b * lengths / self.average_length)
            scores[rows] += self.idf[j] * tf * (self.k1 + 1) / denominator
        return scores


    def top_k(self, query_tokens: List[str],
              k: int) -> List[Tuple[int, float]]:
        """Return the k best documents for a query.

        Args:
            query_tokens: Tokens of the query, from tokenize.
            k: Maximum number of results.

        Returns:
            (document number, score) pairs sorted by decreasing score, ties
            in document order; documents scoring 0 are left out, so the
            list can be shorter than k, and it is empty when k <= 0.
        """
        
        result = []
        if k <= 0:
            return result
        scores = self.score(query_tokens)
        order = np.argsort(-scores, kind="stable")
        for document in order[:k]:
            if scores[document] <= 0:
                break
            result.append((int(document), float(scores[document])))

        return result
