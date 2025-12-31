"""
FastAPI backend exposing MedGuide capabilities.

Endpoints
- GET /health: basic readiness + KB status
- POST /reports/process: upload and process a PDF report, build KB
- GET /reports/final: fetch latest final report text
- GET /reports/pages: list per-page analysis outputs
- POST /chat: ask questions against the knowledge base
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from utils.pdf_extractor import extract_text_from_pdf
from agents.document_extraction_agent import document_extraction_agent
from agents.analyzer_agent import analyzer_agent
from agents.final_report_agent import final_report_agent
from vectordb.create_vector_db import create_vectordb_from_pdfs_and_outputs
from agno.vectordb.search import SearchType
from agents.chat_agent import chat_agent


load_dotenv()

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "knowledge_base" / "outputs"
PDF_TEXT_DIR = DATA_DIR / "knowledge_base" / "pdfs"
UPLOAD_DIR = DATA_DIR / "uploads"
LANCEDB_DIR = DATA_DIR / "lancedb"
TABLE_NAME = "medguide_collection"

for d in [OUTPUT_DIR, PDF_TEXT_DIR, UPLOAD_DIR, LANCEDB_DIR]:
	d.mkdir(parents=True, exist_ok=True)


# ---------- FastAPI app ----------
app = FastAPI(title="MedGuide API", version="0.1.0")


# ---------- Models ----------
class ProcessReportResponse(BaseModel):
	report_name: str
	pages_analyzed: int
	page_outputs: List[str]
	final_report_path: str
	final_report: str
	knowledge_base_ready: bool
	duration_seconds: float


class ChatRequest(BaseModel):
	question: str
	user_id: Optional[str] = "api-user"
	session_id: Optional[str] = "api-session-1"
	use_reranker: bool = True


class ChatResponse(BaseModel):
	answer: str
	retrieved_docs: int


# ---------- Helpers ----------
def _clear_dir(path: Path) -> None:
	if path.exists():
		for item in path.iterdir():
			if item.is_file():
				item.unlink()
			else:
				shutil.rmtree(item)


def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
	dest = dest_dir / upload.filename
	with open(dest, "wb") as f:
		f.write(upload.file.read())
	return dest


async def extract_page_async(page_text: str, page_number: int):
	try:
		prompt = (
			f"Page {page_number} of a blood test report.\n"
			f"Extract all test names, values, and reference ranges as JSON."
		)
		resp = await document_extraction_agent.arun(f"{prompt}\n\n{page_text[:15000]}")
		return resp.content
	except Exception:
		return None


async def analyze_page_async(page_text: str, page_number: int, out_dir: Path):
	try:
		prompt = f"Analyze Page {page_number} of the blood report:\n{page_text[:15000]}"
		resp = await analyzer_agent.arun(prompt)
		output_text = (resp.content or "").strip()
		(out_dir / f"page_{page_number}.txt").write_text(output_text, encoding="utf-8")
		return output_text
	except Exception:
		return f"[Page {page_number}] Analysis failed."


async def run_page_extraction(pdf_path: Path) -> List[str]:
	pages = extract_text_from_pdf(str(pdf_path), by_page=True)
	tasks = [extract_page_async(p, i + 1) for i, p in enumerate(pages)]
	results = await asyncio.gather(*tasks)
	return [r for r in results if r]


async def run_page_analysis(page_texts: List[str], out_dir: Path) -> List[str]:
	tasks = [analyze_page_async(t, i + 1, out_dir) for i, t in enumerate(page_texts)]
	return await asyncio.gather(*tasks)


async def generate_final_report(page_outputs: List[str], out_dir: Path) -> str:
	merged = "\n\n--- PAGE BREAK ---\n\n".join(page_outputs)
	prompt = (
		"You will receive outputs from multiple pages of a blood report. "
		"Combine all pages into a single, structured, user-friendly final health report. "
		"Group related tests into meaningful categories (e.g., Liver Function, Lipid Profile, etc.), "
		"summarize key findings, highlight potential concerns, and provide short, "
		"concise diet and lifestyle recommendations.\n\n"
		"Keep the tone factual, safe, and supportive. Avoid diagnosis or prescriptions.\n\n"
		f"{merged}"
	)
	resp = await final_report_agent.arun(prompt)
	final_text = (resp.content or "").strip()
	(out_dir / "final_report.txt").write_text(final_text, encoding="utf-8")
	return final_text


def _knowledge_ready() -> bool:
	return LANCEDB_DIR.exists() and any(LANCEDB_DIR.glob("**/*.lance"))


_agent = None
_agent_lock = asyncio.Lock()


async def _get_chat_agent(use_reranker: bool):
	global _agent
	async with _agent_lock:
		if _agent is None:
			_agent = chat_agent(
				lancedb_path=str(LANCEDB_DIR),
				collection=TABLE_NAME,
				top_k=5,
				min_docs_for_confident_answer=1,
				use_reranker=use_reranker,
				db_path=str(DATA_DIR / "agent_memory.db"),
				enable_agentic_memory=False,
			)
		return _agent


# ---------- Endpoints ----------
@app.get("/health")
async def health():
	return {
		"status": "ok",
		"knowledge_base_ready": _knowledge_ready(),
		"outputs": len(list(OUTPUT_DIR.glob("*.txt"))),
	}


@app.post("/reports/process", response_model=ProcessReportResponse)
async def process_report(
	upload: UploadFile = File(...),
	recreate_kb: bool = Form(True),
	hybrid: bool = Form(True),
	use_reranker: bool = Form(True),
):
	if upload.content_type not in {"application/pdf", "application/octet-stream"}:
		raise HTTPException(status_code=400, detail="Upload must be a PDF file.")

	start = time.perf_counter()

	_clear_dir(OUTPUT_DIR)
	pdf_path = _save_upload(upload, UPLOAD_DIR)

	try:
		pages = await run_page_extraction(pdf_path)
		page_outputs = await run_page_analysis(pages, OUTPUT_DIR)
		final_text = await generate_final_report(page_outputs, OUTPUT_DIR)

		create_vectordb_from_pdfs_and_outputs(
			base_dir=BASE_DIR,
			pdfs_subdir="data/knowledge_base/pdfs",
			outputs_subdir="data/knowledge_base/outputs",
			lancedb_subdir="data/lancedb",
			recreate=recreate_kb,
			table_name=TABLE_NAME,
			search_type=SearchType.hybrid if hybrid else SearchType.vector,
			use_reranker=use_reranker,
		)

		duration = time.perf_counter() - start
		return ProcessReportResponse(
			report_name=upload.filename,
			pages_analyzed=len(page_outputs),
			page_outputs=[p.name for p in sorted(OUTPUT_DIR.glob("page_*.txt"))],
			final_report_path=str(OUTPUT_DIR / "final_report.txt"),
			final_report=final_text,
			knowledge_base_ready=_knowledge_ready(),
			duration_seconds=duration,
		)
	except Exception as exc:  # pylint: disable=broad-except
		raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc


@app.get("/reports/final", response_class=PlainTextResponse)
async def get_final_report():
	final_path = OUTPUT_DIR / "final_report.txt"
	if not final_path.exists():
		raise HTTPException(status_code=404, detail="No final report found. Process a PDF first.")
	return final_path.read_text(encoding="utf-8")


@app.get("/reports/pages")
async def list_page_outputs():
	files = sorted(OUTPUT_DIR.glob("page_*.txt"))
	if not files:
		raise HTTPException(status_code=404, detail="No page analyses found. Process a PDF first.")
	return {"pages": [f.name for f in files]}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
	if not _knowledge_ready():
		raise HTTPException(status_code=400, detail="Knowledge base not ready. Process a PDF first.")

	agent = await _get_chat_agent(body.use_reranker)
	answer_text, retrieved_count = await asyncio.to_thread(
		agent.answer,
		body.question,
		user_id=body.user_id,
		session_id=body.session_id,
	)
	return ChatResponse(answer=answer_text, retrieved_docs=retrieved_count)


@app.get("/")
async def root():
	return {"message": "MedGuide API is running. See /docs for OpenAPI."}
