# main.py — Page-wise Parallel Analysis + Final Report
import os
import json
import asyncio
import shutil
from pathlib import Path
from dotenv import load_dotenv
from time import perf_counter

from utils.pdf_extractor import extract_text_from_pdf
from agents.document_extraction_agent import document_extraction_agent
from agents.analyzer_agent import analyzer_agent
from agents.final_report_agent import final_report_agent
from vectordb.create_vector_db import create_vectordb_from_pdfs_and_outputs
from agents.chat_agent import chat_agent


# -------- Config --------
load_dotenv()
PARALLEL_TIMEOUT = 200

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "knowledge_base" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 🧹 Clean up existing files before every run
if any(OUTPUT_DIR.iterdir()):
    print(f"🧹 Cleaning up old files in {OUTPUT_DIR}...")
    shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Async Step 1: Page Extraction --------
async def extract_page_async(page_text: str, page_number: int):
    """Run the document extraction agent for a single page."""
    try:
        prompt = (
            f"Page {page_number} of a blood test report.\n"
            f"Extract all test names, values, and reference ranges as JSON."
        )
        resp = await document_extraction_agent.arun(f"{prompt}\n\n{page_text[:15000]}")
        return resp.content
    except Exception as e:
        print(f"⚠️ Page {page_number} extraction failed: {e}")
        return None


async def extract_blood_report_async(pdf_path: str):
    """Extract text page-wise and run extraction agents concurrently."""
    pdf_file_path = Path(pdf_path)
    if not pdf_file_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"📄 Extracting text per page from PDF: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path, by_page=True)
    print(f"✅ Extracted {len(pages)} pages")

    # Run all page extractions in parallel
    tasks = [extract_page_async(p, i + 1) for i, p in enumerate(pages)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]


# -------- Async Step 2: Page Analysis (Parallel) --------
async def analyze_page_async(page_text: str, page_number: int):
    """Run the combined page analysis agent for a single page."""
    try:
        prompt = f"Analyze Page {page_number} of the blood report:\n{page_text[:15000]}"
        resp = await analyzer_agent.arun(prompt)
        output_text = resp.content.strip()

        # Save individual page output
        page_file = OUTPUT_DIR / f"page_{page_number}.txt"
        page_file.write_text(output_text, encoding="utf-8")

        print(f"✅ Saved page {page_number} analysis → {page_file}")
        return output_text

    except Exception as e:
        print(f"⚠️ Page {page_number} analysis failed: {e}")
        return f"[Page {page_number}] Analysis failed."


async def run_parallel_analysis(page_texts: list[str]):
    """Run the analyzer+risk logic in parallel for all pages."""
    print("🚀 Running page-wise analysis in parallel...")
    tasks = [analyze_page_async(t, i + 1) for i, t in enumerate(page_texts)]
    results = await asyncio.gather(*tasks)
    print(f"✅ Completed {len(results)} page analyses")
    return results


# -------- Async Step 3: Final Report --------
async def generate_final_report(page_outputs: list[str]):
    """Combine all page-level analyses into one final report."""
    try:
        merged_text = "\n\n--- PAGE BREAK ---\n\n".join(page_outputs)
        prompt = (
            "You will receive outputs from multiple pages of a blood report. "
            "Combine all pages into a single, structured, user-friendly final health report. "
            "Group related tests into meaningful categories (e.g., Liver Function, Lipid Profile, etc.), "
            "summarize key findings, highlight potential concerns, and provide short, "
            "concise diet and lifestyle recommendations.\n\n"
            "Keep the tone factual, safe, and supportive. Avoid diagnosis or prescriptions.\n\n"
            f"{merged_text}"
        )

        print("🧩 Synthesizing final report...")
        resp = await final_report_agent.arun(prompt)
        final_text = resp.content.strip()

        # Save the combined final report
        final_path = OUTPUT_DIR / "final_report.txt"
        final_path.write_text(final_text, encoding="utf-8")
        print(f"💾 Final report saved → {final_path}")

        return final_text

    except Exception as e:
        print(f"❌ Final report generation failed: {e}")
        return None

def always_clear_memory(db_path: str):
    """
    Always clears the agent's memory or vector DB folder before each run.
    Works for both file and directory paths.
    """
    try:
        if os.path.exists(db_path):
            if os.path.isdir(db_path):
                shutil.rmtree(db_path)
            else:
                os.remove(db_path)
            print(f"🧹 Automatically cleared memory at: {db_path}")
    except Exception as e:
        print(f"⚠️ Could not clear memory: {e}")

# -------- Pipeline --------
async def run_pipeline(pdf_path: str):
    start = perf_counter()

    # 1️⃣ Extract Text & Basic Data
    pages = await extract_blood_report_async(pdf_path)

    # 2️⃣ Analyze Each Page in Parallel
    page_outputs = await run_parallel_analysis(pages)

    # 3️⃣ Final Report Generation
    final_report_text = await generate_final_report(page_outputs)

    # 4️⃣ Output
    if final_report_text:
        print("\n================= 🩺 FINAL HEALTH REPORT =================\n")
        print(final_report_text)

    print(f"\n⏱️ Total Time: {perf_counter() - start:.1f}s")


# -------- Entry Point --------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(BASE_DIR, "data", "sample_reports", "labreportnew.pdf")

    try:
        asyncio.run(run_pipeline(pdf_path))
    except Exception as e:
        print(f"❌ Error: {e}")

    # vectordb/create_vector_db.py
    create_vectordb_from_pdfs_and_outputs()


    db_path = "data/agent_memory.db"
    always_clear_memory(db_path)

    # Create the agent only once (using LanceDB now)
    agent = chat_agent(
        lancedb_path="data/lancedb",   # ✅ replaced Chroma with LanceDB
        collection="medguide_collection",
        top_k=5,
        min_docs_for_confident_answer=1,
        use_reranker=True,
        db_path=db_path,
        enable_agentic_memory=False,
    )

    # Reuse the same agent for multiple queries
    queries = [
        "Interpret Glycosylated Hemoglobin results in my report and suggest possible causes.",
        "Interpret ESTIMATED AVG. GLUCOSE results in my report and suggest possible causes.",
        "WHAT WAS THE QUESTION THAT I ASKED PREVIOUS TO PREVIOUS, JUST GIVE ME THE QUESTION WORD BY WORD.",
        "what was the question that i asked previously, just give me the question word by word."
    ]

    for q in queries:
        answer_text, retrieved_count = agent.answer(
            q,
            user_id="user-123",
            session_id="medguide-session-1",
        )

        print(f"\n📄 Retrieved docs: {retrieved_count}")
        print(f"\n🧠 Final Answer:\n{answer_text}")
    