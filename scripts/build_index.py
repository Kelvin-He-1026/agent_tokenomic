"""Build the frozen Chroma index the retrieve tool reads.

Controlled variable, built once and never rebuilt mid-experiment: the corpus,
embedding model, chunking and top-K all stay fixed while runtime, architecture
and model vary. The manifest records content hashes so a later rebuild can be
shown to be identical.

Embeddings load from model/<embedding_local_dir>, never from the Hub.

    python scripts/build_index.py --config configs/pilot_qwen.yaml \
        --documents ../agentic_ai/documents
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import RunConfig  # noqa: E402

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_documents(doc_dir: Path):
    """Read PDFs with pypdf directly rather than through langchain-community,
    which is sunset. One Document per page, source and page recorded."""
    from langchain_core.documents import Document
    from pypdf import PdfReader

    docs: list[Document] = []
    pdfs = sorted(doc_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs in {doc_dir}")
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        kept = 0
        for page_no, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue  # image-only page; nothing to embed
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(pdf), "page": page_no},
                )
            )
            kept += 1
        print(f"  {pdf.name}: {kept}/{len(reader.pages)} pages with text  sha={file_hash(pdf)}")
    return docs, {p.name: file_hash(p) for p in pdfs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--documents", required=True)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    chroma_dir = cfg.resolve(cfg.chroma_dir)
    embed_path = cfg.embedding_path()

    if not embed_path.exists():
        raise SystemExit(f"embedding model missing at {embed_path}; run scripts/fetch_models.py")
    if chroma_dir.exists() and not args.rebuild:
        raise SystemExit(f"{chroma_dir} exists; pass --rebuild to replace it")

    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"loading documents from {args.documents}")
    docs, hashes = load_documents(Path(args.documents))

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")

    print(f"embedding with {embed_path}")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name=str(embed_path)),
        persist_directory=str(chroma_dir),
        collection_name=cfg.chroma_collection,
    )
    count = store._collection.count()

    manifest = chroma_dir.parent / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "collection": cfg.chroma_collection,
                "embedding_model": cfg.embedding_model,
                "embedding_path": str(embed_path),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "top_k": cfg.retrieval_top_k,
                "n_chunks": count,
                "documents": hashes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nindexed {count} chunks -> {chroma_dir}")
    print(f"manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
