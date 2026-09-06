"""
FAISS vector store management.

Responsibilities:
  - Build a FAISS index from CV chunks + embeddings.
  - Persist index to disk with a companion index_meta.json that captures:
      cv_hash, embedding_model, chunk_size, chunk_overlap
  - On startup, compare the current CV file's SHA-256 hash and chunking
    settings against the stored metadata; rebuild automatically when stale.
  - Expose the raw FAISS store (not just a retriever) so cv_search.py can
    use similarity_search_with_score for relevance gating.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from cv_agent.config import settings

logger = logging.getLogger(__name__)

_META_FILENAME = "index_meta.json"


# ── CV hashing ────────────────────────────────────────────────────────────────

def compute_cv_hash(path: Path | list[Path] | tuple[Path, ...]) -> str:
    """Return the combined SHA-256 hex digest of one or multiple CV files."""
    sha256 = hashlib.sha256()
    paths = [path] if isinstance(path, Path) else sorted(path, key=lambda p: str(p))
    for p in paths:
        if p.exists() and p.is_file():
            sha256.update(p.name.encode("utf-8"))
            with open(p, "rb") as fh:
                for block in iter(lambda: fh.read(8192), b""):
                    sha256.update(block)
    digest = sha256.hexdigest()
    logger.debug("INDEX | CV(s) hash computed: %s", digest[:16] + "…")
    return digest


# ── Index metadata ────────────────────────────────────────────────────────────

def _read_meta(index_path: Path) -> dict:
    """Load stored index metadata, or return empty dict if not found."""
    meta_file = index_path / _META_FILENAME
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("INDEX | Failed to read index_meta.json: %s", exc)
    return {}


def _write_meta(index_path: Path, cv_hash: str) -> None:
    """Persist index metadata alongside the FAISS index directory."""
    meta = {
        "cv_hash": cv_hash,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    meta_file = index_path / _META_FILENAME
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("INDEX | Metadata written to %s", meta_file)


def _index_is_stale(index_path: Path, cv_hash: str) -> bool:
    """
    Return True if the stored index doesn't match the current CV / config.

    Checks: cv_hash, embedding_model, chunk_size, chunk_overlap.
    Any mismatch triggers a full rebuild.
    """
    meta = _read_meta(index_path)
    if not meta:
        logger.info("INDEX | No metadata found — treating index as stale")
        return True

    checks = {
        "cv_hash": cv_hash,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    for key, expected in checks.items():
        stored = meta.get(key)
        if stored != expected:
            logger.info(
                "INDEX | Stale: %s changed | stored=%r -> current=%r",
                key,
                stored,
                expected,
            )
            return True

    logger.info("INDEX | Metadata matches current CV and config — index is fresh")
    return False


# ── Embedding model factory ───────────────────────────────────────────────────

def _get_embeddings():
    """Return the configured embedding model."""
    provider = settings.llm_provider

    if provider == "gemini":
        if not settings.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Please set GOOGLE_API_KEY in your .env file "
                "or set it as an environment variable."
            )
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is required. "
                "Install with: pip install langchain-google-genai"
            ) from exc
        logger.debug("INDEX | Using Google Generative AI embeddings: %s", settings.embedding_model)
        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )

    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Please set OPENAI_API_KEY in your .env file "
                "or set it as an environment variable."
            )
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required. "
                "Install with: pip install langchain-openai"
            ) from exc
        logger.debug("INDEX | Using OpenAI embeddings: %s", settings.embedding_model)
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    elif provider == "ollama":
        try:
            from langchain_community.embeddings import OllamaEmbeddings
        except ImportError as exc:
            raise ImportError(
                "langchain-community is required for Ollama embeddings."
            ) from exc
        logger.debug("INDEX | Using Ollama embeddings: %s", settings.embedding_model)
        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"Unknown llm_provider: {provider!r}")


def _build_faiss_with_retry(chunks: list[Document], embeddings) -> FAISS:
    """Build FAISS index in batches with progressive backoff on 429 rate limits."""
    batch_size = 10
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    all_embeddings: list[list[float]] = []
    delays = [3.0, 6.0, 12.0, 20.0, 30.0]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        logger.info(
            "INDEX | Embedding batch %d..%d of %d chunks",
            i + 1,
            i + len(batch_texts),
            len(texts),
        )

        for attempt in range(1, len(delays) + 1):
            try:
                batch_emb = embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_emb)
                break
            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    delay = delays[attempt - 1]
                    logger.warning(
                        "INDEX | Rate limited (429) on batch %d..%d — waiting %.1fs (attempt %d/%d)...",
                        i + 1,
                        i + len(batch_texts),
                        delay,
                        attempt,
                        len(delays),
                    )
                    time.sleep(delay)
                else:
                    logger.error("INDEX | Embedding batch failed: %s", exc)
                    raise exc
        else:
            raise RuntimeError(
                f"Failed to embed document batch {i+1}..{i+len(batch_texts)} after {len(delays)} retries due to quota limits."
            )

        if i + batch_size < len(texts):
            time.sleep(0.3)

    text_embedding_pairs = list(zip(texts, all_embeddings))
    return FAISS.from_embeddings(text_embedding_pairs, embeddings, metadatas=metadatas)


# ── Public API ────────────────────────────────────────────────────────────────

def build_or_load_index(
    chunks: list[Document] | None = None,
    cv_path: Path | list[Path] | None = None,
) -> FAISS:
    """
    Return a FAISS vector store, loading from disk when the index is fresh
    or rebuilding when the CV file(s) / configuration has changed.

    Logic:
        1. If the index directory exists AND metadata matches the current CV(s)
           hash + config -> load from disk.
        2. Otherwise -> build from *chunks* and persist with updated metadata.

    Args:
        chunks:  Pre-chunked Documents from the ingestion module.
                 Required when a rebuild is triggered.
        cv_path: Path or list of Paths to CV files used for SHA-256 hashing.
                 Defaults to settings.cv_file_path.

    Returns:
        A ready FAISS vector store instance.
    """
    index_path = settings.faiss_index_path
    embeddings = _get_embeddings()
    target_paths = cv_path if cv_path is not None else settings.cv_file_path

    # ── Compute current CV hash ───────────────────────────────────────────────
    cv_hash = compute_cv_hash(target_paths)

    # ── Try loading existing index ────────────────────────────────────────────
    if index_path.exists() and not _index_is_stale(index_path, cv_hash):
        logger.info("INDEX | Loading existing FAISS index from '%s'", index_path)
        t0 = time.perf_counter()
        try:
            store = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("INDEX | FAISS index loaded in %.2fs", time.perf_counter() - t0)
            return store
        except Exception as exc:
            logger.warning(
                "INDEX | Failed to load existing index (%s) — rebuilding", exc
            )

    # ── Build new index ───────────────────────────────────────────────────────
    if chunks is None:
        raise ValueError(
            "No chunks provided and no valid FAISS index found at "
            f"'{index_path}'. Run ingestion first."
        )

    logger.info(
        "INDEX | Building FAISS index from %d chunk(s) | model=%s",
        len(chunks),
        settings.embedding_model,
    )
    t0 = time.perf_counter()
    try:
        store = _build_faiss_with_retry(chunks, embeddings)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build FAISS index: {exc}. "
            "Check your API key and network connection."
        ) from exc

    elapsed = time.perf_counter() - t0
    logger.info("INDEX | FAISS index built in %.2fs", elapsed)

    # ── Persist index + metadata ──────────────────────────────────────────────
    index_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_path))
    logger.info("INDEX | FAISS index persisted to '%s'", index_path)

    if cv_hash:
        _write_meta(index_path, cv_hash)

    return store


def get_retriever(store: FAISS):
    """Return a LangChain retriever configured with top-k from settings."""
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )
