"""Discovery and reading of the raw corpus files."""

from pathlib import Path
from typing import Optional, List, Dict
from src.models import ChunkKind


EXTENSION_KINDS: Dict[str, ChunkKind] = {".py": "python",
                                         ".md": "markdown",
                                         ".txt": "text",
                                         ".yaml": "text",
                                         ".yml": "text",
                                         ".toml": "text",
                                         ".sh": "text",
                                         ".cmake": "text",
                                         ".cfg": "text",
                                         ".ini": "text",
                                         ".cu": "text",
                                         ".cuh": "text",
                                         ".cpp": "text",
                                         ".h": "text",
                                         ".hpp": "text",
                                         ".jinja": "text"
                                         }


def kind_for_path(path: str) -> Optional[ChunkKind]:
    """Return the chunking strategy for a file, based on its extension.

    Args:
        path: Path of the file; only its extension is used,
            case-insensitively.

    Returns:
        "python", "markdown" or "text", or None if the file is not indexed.
    """
    extension = Path(path).suffix.lower()
    return EXTENSION_KINDS.get(extension)


def list_corpus_files(corpus_dir: str) -> List[str]:
    """List the indexable files under the corpus directory.

    Args:
        corpus_dir: Directory to scan recursively.

    Returns:
        Normalised POSIX paths of the files with a supported extension,
        sorted so that chunk order is reproducible between builds.

    Raises:
        FileNotFoundError: If corpus_dir is not an existing directory.
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"The path doesn't exist {corpus_dir}")

    paths = []
    for x in root.rglob("*"):
        if kind_for_path(str(x)):
            if x.is_file():
                paths.append(x.as_posix())

    return (sorted(paths))


def read_text_file(path: str) -> Optional[str]:
    """Read a corpus file as UTF-8 text.

    Args:
        path: Path of the file to read.

    Returns:
        The file content, or None if it is not valid UTF-8.
    """

    try:
        with open(path, encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        return None
