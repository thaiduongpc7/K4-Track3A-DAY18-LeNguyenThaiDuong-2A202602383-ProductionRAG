from __future__ import annotations

"""Module 1: Advanced Chunking Strategies."""

import glob
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)

_SEMANTIC_MODEL = None
_SEMANTIC_MODEL_UNAVAILABLE = False


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract the text layer from a PDF."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load Markdown and text-layer PDF documents from data_dir."""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        try:
            text = _extract_pdf_text(fp)
        except ImportError:
            print(
                f"  Warning: skipped {os.path.basename(fp)} because pypdf is not installed."
            )
            continue
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(
                f"  Warning: skipped {os.path.basename(fp)} because it has no text layer."
            )

    return docs


def chunk_basic(
    text: str, chunk_size: int = 500, metadata: dict | None = None
) -> list[Chunk]:
    """Basic paragraph-based chunking used as a comparison baseline."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    metadata = dict(metadata or {})
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) > chunk_size and current:
            chunks.append(
                Chunk(
                    text=current.strip(),
                    metadata={**metadata, "chunk_index": len(chunks)},
                )
            )
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        chunks.append(
            Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_index": len(chunks)},
            )
        )
    return chunks


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text into groups of adjacent sentences using cosine similarity.

    The primary path uses all-MiniLM-L6-v2. A lexical fallback keeps local
    tests and offline ingestion usable if the model cannot be loaded.
    """
    metadata = dict(metadata or {})
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\n", text)
        if sentence.strip()
    ]
    if not sentences:
        return []

    try:
        import numpy as np
        model = _get_semantic_model()
        if model is None:
            raise RuntimeError("semantic model is unavailable")
        embeddings = np.asarray(model.encode(sentences))
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-9)
        similarities = np.sum(normalized[:-1] * normalized[1:], axis=1).tolist()
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        similarities = [
            _lexical_cosine_similarity(sentences[index - 1], sentences[index])
            for index in range(1, len(sentences))
        ]

    groups: list[list[str]] = [[sentences[0]]]
    for index, sentence in enumerate(sentences[1:]):
        if similarities[index] < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(
            text=" ".join(group),
            metadata={**metadata, "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
    ]


def _get_semantic_model():
    global _SEMANTIC_MODEL, _SEMANTIC_MODEL_UNAVAILABLE
    if _SEMANTIC_MODEL is not None or _SEMANTIC_MODEL_UNAVAILABLE:
        return _SEMANTIC_MODEL

    try:
        from sentence_transformers import SentenceTransformer

        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except (ImportError, OSError, RuntimeError, ValueError):
        _SEMANTIC_MODEL_UNAVAILABLE = True
    return _SEMANTIC_MODEL


def _lexical_cosine_similarity(first: str, second: str) -> float:
    """Deterministic fallback used only when sentence embeddings are unavailable."""
    token_pattern = r"\w+"
    first_counts = Counter(re.findall(token_pattern, first.casefold()))
    second_counts = Counter(re.findall(token_pattern, second.casefold()))
    if not first_counts or not second_counts:
        return 0.0

    common = set(first_counts) & set(second_counts)
    numerator = sum(first_counts[token] * second_counts[token] for token in common)
    first_norm = sum(value * value for value in first_counts.values()) ** 0.5
    second_norm = sum(value * value for value in second_counts.values()) ** 0.5
    return numerator / (first_norm * second_norm + 1e-9)


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Build parent chunks for context and child chunks for retrieval.

    Returns:
        (parents, children), where each child has a parent_id field and the
        same parent_id in its metadata.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")

    metadata = dict(metadata or {})
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(_split_text_by_size(paragraph, parent_size))

    parents: list[Chunk] = []
    current_units: list[str] = []
    current_length = 0
    for unit in units:
        separator_length = 2 if current_units else 0
        if current_units and current_length + separator_length + len(unit) > parent_size:
            parents.append(_make_parent_chunk(current_units, metadata, len(parents)))
            current_units = []
            current_length = 0
        current_units.append(unit)
        current_length += (2 if len(current_units) > 1 else 0) + len(unit)
    if current_units:
        parents.append(_make_parent_chunk(current_units, metadata, len(parents)))

    children: list[Chunk] = []
    for parent in parents:
        parent_id = parent.metadata["parent_id"]
        for child_text in _split_text_by_size(parent.text, child_size):
            child_metadata = {
                **metadata,
                "chunk_type": "child",
                "strategy": "hierarchical",
                "parent_id": parent_id,
                "chunk_index": len(children),
            }
            children.append(
                Chunk(
                    text=child_text,
                    metadata=child_metadata,
                    parent_id=parent_id,
                )
            )

    return parents, children


def _split_text_by_size(text: str, max_size: int) -> list[str]:
    """Split text without creating a piece larger than max_size."""
    text = text.strip()
    if not text:
        return []

    pieces: list[str] = []
    remaining = text
    while len(remaining) > max_size:
        boundary = max(
            remaining.rfind(" ", 0, max_size + 1),
            remaining.rfind("\n", 0, max_size + 1),
            remaining.rfind("\t", 0, max_size + 1),
        )
        if boundary <= 0:
            boundary = max_size
        piece = remaining[:boundary].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[boundary:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _make_parent_chunk(units: list[str], metadata: dict, index: int) -> Chunk:
    parent_id = f"parent_{index}"
    parent_metadata = {
        **metadata,
        "chunk_type": "parent",
        "strategy": "hierarchical",
        "parent_id": parent_id,
        "chunk_index": index,
    }
    return Chunk(
        text="\n\n".join(units),
        metadata=parent_metadata,
        parent_id=parent_id,
    )


def chunk_structure_aware(
    text: str, metadata: dict | None = None
) -> list[Chunk]:
    """
    Split Markdown into sections headed by #, ##, or ###.

    Heading-looking lines inside fenced code blocks are treated as ordinary
    content so code blocks remain intact.
    """
    metadata = dict(metadata or {})
    heading_pattern = re.compile(r"^(#{1,3})[ \t]+.+?\s*$")
    sections: list[tuple[str, str]] = []
    current_header = ""
    current_lines: list[str] = []
    in_fenced_block = False

    def flush_section() -> None:
        nonlocal current_lines
        section_text = "".join(current_lines).strip()
        if section_text:
            sections.append((current_header, section_text))
        current_lines = []

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        fence_match = re.match(r"^(```|~~~)", stripped)
        heading_match = (
            None
            if in_fenced_block
            else heading_pattern.match(line.rstrip("\r\n"))
        )

        if heading_match:
            flush_section()
            current_header = heading_match.group(0).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

        if fence_match:
            in_fenced_block = not in_fenced_block

    flush_section()

    return [
        Chunk(
            text=section_text,
            metadata={
                **metadata,
                "section": section,
                "strategy": "structure",
                "chunk_index": index,
            },
        )
        for index, (section, section_text) in enumerate(sections)
    ]


def compare_strategies(documents: list[dict]) -> dict:
    """Run all strategies on documents and print simple size statistics."""

    def _stats(chunk_list):
        lengths = [len(chunk.text) for chunk in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(document["text"] for document in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, stats in results.items():
        print(
            f"{name:<15} {stats['count']:>7} {stats['avg_len']:>5} "
            f"{stats['min_len']:>5} {stats['max_len']:>5}"
        )

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
