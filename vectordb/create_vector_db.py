import os
import shutil
import pathlib
from typing import Optional

from dotenv import load_dotenv
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.reranker.cohere import CohereReranker
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.knowledge.reader.text_reader import TextReader

load_dotenv()


def create_vectordb_from_pdfs_and_outputs(
    base_dir: Optional[str | pathlib.Path] = None,
    pdfs_subdir: str = "data/knowledge_base/pdfs",
    outputs_subdir: str = "data/knowledge_base/outputs",
    lancedb_subdir: str = "data/lancedb",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    recreate: bool = True,
    table_name: str = "medguide_collection",
    search_type: SearchType = SearchType.vector,  # can be vector / keyword / hybrid
    use_reranker: bool = True,
) -> Knowledge:
    """
    Build a persistent LanceDB knowledge base from .txt files in:
      - data/knowledge_base/pdfs
      - data/knowledge_base/outputs

    Uses:
      - OpenAI embeddings (text-embedding-3-small)
      - Cohere reranker (optional)
      - Agno TextReader for chunking
      - LanceDB for storage and retrieval
    """

    # -------- Paths setup --------
    base_dir = pathlib.Path(base_dir or pathlib.Path(__file__).resolve().parent.parent)
    pdfs_dir = base_dir / pdfs_subdir
    outputs_dir = base_dir / outputs_subdir
    lancedb_dir = base_dir / lancedb_subdir

    if recreate and lancedb_dir.exists():
        print(f"🧹 Removing old LanceDB directory at: {lancedb_dir}")
        shutil.rmtree(lancedb_dir)
    lancedb_dir.mkdir(parents=True, exist_ok=True)

    # -------- Collect .txt files --------
    pdf_txt_files = [f for f in pdfs_dir.glob("*.txt") if f.is_file()]
    output_txt_files = [f for f in outputs_dir.glob("*.txt") if f.is_file()]

    if not pdf_txt_files and not output_txt_files:
        raise FileNotFoundError("❌ No .txt files found in pdfs/ or outputs/")

    print(f"📚 Found {len(pdf_txt_files)} PDF text files and {len(output_txt_files)} output text files")

    # -------- Initialize LanceDB Vector Store --------
    vector_db = LanceDb(
        uri=str(lancedb_dir),
        table_name=table_name,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        search_type=search_type.hybrid,
        reranker=CohereReranker(model="rerank-multilingual-v3.0") if use_reranker else None,
    )

    # -------- Create Knowledge base --------
    knowledge = Knowledge(vector_db=vector_db)

    # -------- Reader for chunking --------
    reader = TextReader(
        chunk=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # -------- Ingest files (with metadata) --------
    print("🧠 Adding documents to LanceDB vector store...")

    count = 0
    for path in pdf_txt_files:
        knowledge.add_content(
            path=str(path),
            reader=reader,
            metadata={"source": "pdf", "file_name": path.name},
        )
        count += 1

    for path in output_txt_files:
        knowledge.add_content(
            path=str(path),
            reader=reader,
            metadata={"source": "output", "file_name": path.name},
        )
        count += 1

    print(f"✅ Ingested {count} files into LanceDB table '{table_name}' at {lancedb_dir}")
    print("✅ Vector database created successfully (LanceDB).")

    return knowledge

