"""Build the public demo knowledge base from the sample data in demo/.

The full data/ folder is git-ignored, so a fresh deployment starts with no
indexes. This module seeds data/ with the bundled sample Q&A pairs, then runs
the normal ingestion and indexing steps unchanged.

demo/document_chunks.csv is optional. The public repo bundles Q&A pairs only,
so no verbatim document text is published; without it, an empty passage index
is written so the retriever starts and the passage layer simply finds nothing.

Run manually with: python -m src.demo.bootstrap
"""
import json
import shutil
from pathlib import Path

import faiss

from src.ingestion.load_dataset import load_dataset
from src.retrieval.build_chunk_index import build_chunk_index
from src.retrieval.build_index import build_index
from src.retrieval.model_cache import get_model

DEMO_DIR = Path("demo")
DEMO_QA_PATH = DEMO_DIR / "qa_pairs.csv"
DEMO_CHUNKS_PATH = DEMO_DIR / "document_chunks.csv"

RAW_QA_PATH = Path("data/raw/demo_qa_pairs.csv")
CHUNKS_PATH = Path("data/processed/document_chunks.csv")
CHUNK_INDEX_PATH = Path("data/processed/chunk_index.faiss")
CHUNK_META_PATH = Path("data/processed/chunk_meta.json")
INDEX_PATHS = [
    Path("data/processed/index.faiss"),
    Path("data/processed/meta.json"),
    CHUNK_INDEX_PATH,
    CHUNK_META_PATH,
]


def indexes_exist() -> bool:
    return all(p.exists() for p in INDEX_PATHS)


def _write_empty_chunk_index() -> None:
    dimension = get_model().get_sentence_embedding_dimension()
    CHUNK_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss.IndexFlatIP(dimension), str(CHUNK_INDEX_PATH))
    CHUNK_META_PATH.write_text(json.dumps([]), encoding="utf-8")


def bootstrap_demo(force: bool = False) -> bool:
    """Seed data/ from demo/ and build both indexes.

    Returns True if indexes were built, False if they already existed.
    """
    if indexes_exist() and not force:
        return False

    if not DEMO_QA_PATH.exists():
        raise FileNotFoundError(f"Demo data file not found: {DEMO_QA_PATH}")

    RAW_QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEMO_QA_PATH, RAW_QA_PATH)
    load_dataset()
    build_index()

    if DEMO_CHUNKS_PATH.exists():
        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEMO_CHUNKS_PATH, CHUNKS_PATH)
        build_chunk_index()
    else:
        _write_empty_chunk_index()
    return True


if __name__ == "__main__":
    built = bootstrap_demo(force=True)
    print("Demo knowledge base built." if built else "Demo indexes already present.")
