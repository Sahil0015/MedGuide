import os
import io
import sys
import base64
import asyncio
import pathlib
from time import perf_counter
from typing import List
import time
import streamlit as st
from dotenv import load_dotenv

# --- Local imports ---
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.pdf_extractor import extract_text_from_pdf
from agents.document_extraction_agent import document_extraction_agent
from agents.analyzer_agent import analyzer_agent
from agents.final_report_agent import final_report_agent
from vectordb.create_vector_db import create_vectordb_from_pdfs_and_outputs
from agno.vectordb.search import SearchType
from agents.chat_agent import chat_agent
from agno.models.openai import OpenAIChat

# ---------- Load env ----------
load_dotenv()

# ---------- App Config ----------
st.set_page_config(page_title="MedGuide: AI Health Companion", page_icon="🩺", layout="centered")

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "knowledge_base" / "outputs"
UPLOAD_DIR = DATA_DIR / "uploads"
LANCEDB_DIR = DATA_DIR / "lancedb"
TABLE_NAME = "medguide_collection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LANCEDB_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Session-state defaults ----------
defaults = {
    "initialized": False,
    "processed": False,
    "processed_file": None,
    "final_text": "",
    "final_displayed": False,  # <-- NEW
    "kb_ready": False,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---------- Helpers ----------
def save_uploaded_file(uploaded_file, dest_dir: pathlib.Path) -> pathlib.Path:
    dest = dest_dir / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.read())
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


async def analyze_page_async(page_text: str, page_number: int, out_dir: pathlib.Path):
    try:
        prompt = f"Analyze Page {page_number} of the blood report:\n{page_text[:15000]}"
        resp = await analyzer_agent.arun(prompt)
        output_text = (resp.content or "").strip()
        (out_dir / f"page_{page_number}.txt").write_text(output_text, encoding="utf-8")
        return output_text
    except Exception:
        return f"[Page {page_number}] Analysis failed."


async def run_page_extraction(pdf_path: pathlib.Path) -> List[str]:
    pages = extract_text_from_pdf(str(pdf_path), by_page=True)
    tasks = [extract_page_async(p, i + 1) for i, p in enumerate(pages)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]


async def run_page_analysis(page_texts: List[str], out_dir: pathlib.Path) -> List[str]:
    tasks = [analyze_page_async(t, i + 1, out_dir) for i, t in enumerate(page_texts)]
    return await asyncio.gather(*tasks)


async def generate_final_report(page_outputs: List[str], out_dir: pathlib.Path) -> str:
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


def make_data_link(label: str, text: str, filename: str) -> str:
    b64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}" target="_blank">{label}</a>'


async def stream_markdown_smooth(text: str, delay: float = 0.15):
    """Stream markdown text paragraph by paragraph for natural display."""
    paragraphs = text.split("\n\n")
    placeholder = st.empty()
    streamed_text = ""

    for para in paragraphs:
        for i in range(0, len(para), 200):  # ~40–50 words per burst
            streamed_text += para[i:i+200]
            placeholder.markdown(streamed_text)
            time.sleep(delay)
        streamed_text += "\n\n"
        placeholder.markdown(streamed_text)
        time.sleep(0.2)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Configuration")
    model_id = st.text_input("Model id", value="gpt-4o-mini")
    openai_key = st.text_input("OPENAI_API_KEY", type="password")
    enable_reranker = st.toggle("Use Cohere reranker", value=True)
    cohere_key = st.text_input("COHERE_API_KEY", type="password") if enable_reranker else ""
    hybrid = st.toggle("Hybrid retrieval (LanceDB BM25+vector)", value=True)

    col1, col2 = st.columns(2)
    with col1:
        init_clicked = st.button("Initialize")
    with col2:
        reset_clicked = st.button("Reset")

    if init_clicked:
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if enable_reranker and cohere_key:
            os.environ["COHERE_API_KEY"] = cohere_key
        st.session_state["initialized"] = True
        st.success("Initialized. Ready to process a PDF.")

    if reset_clicked:
        for key in ["processed", "processed_file", "final_text", "kb_ready", "final_displayed"]:
            st.session_state[key] = False if key == "processed" else None
        st.experimental_rerun()

st.title("🩺 MedGuide: AI Health Companion")

if not st.session_state.get("initialized"):
    st.info("Please configure keys and model in the sidebar, then click Initialize.")
    st.stop()

# ---------- Upload + Process ----------
uploaded = st.file_uploader("Upload a lab report (PDF)", type=["pdf"])
process_clicked = st.button("Process report")

if uploaded is not None and st.session_state.get("processed_file") != uploaded.name:
    st.session_state["processed"] = False
    st.session_state["processed_file"] = uploaded.name
    st.session_state["final_displayed"] = False

if uploaded and (process_clicked or st.session_state["processed"]):
    # Process new PDF
    if process_clicked and not st.session_state["processed"]:
        start = perf_counter()
        # clear old outputs
        for p in OUTPUT_DIR.glob("*"):
            if p.is_file():
                p.unlink()
            else:
                import shutil
                shutil.rmtree(p)

        pdf_path = save_uploaded_file(uploaded, UPLOAD_DIR)
        with st.spinner("Extracting pages..."):
            pages = asyncio.run(run_page_extraction(pdf_path))
        with st.spinner("Analyzing pages..."):
            page_outputs = asyncio.run(run_page_analysis(pages, OUTPUT_DIR))
        with st.spinner("Synthesizing final report..."):
            final_text = asyncio.run(generate_final_report(page_outputs, OUTPUT_DIR))

        st.session_state["final_text"] = final_text
        st.session_state["processed"] = True
        st.session_state["final_displayed"] = False

        with st.spinner("Building knowledge base..."):
            create_vectordb_from_pdfs_and_outputs(
                base_dir=BASE_DIR,
                pdfs_subdir="data/knowledge_base/pdfs",
                outputs_subdir="data/knowledge_base/outputs",
                lancedb_subdir="data/lancedb",
                recreate=True,
                table_name=TABLE_NAME,
                search_type=SearchType.hybrid if hybrid else SearchType.vector,
                use_reranker=bool(enable_reranker and cohere_key),
            )
        st.session_state["kb_ready"] = True
        st.success(f"Report processed successfully in {perf_counter() - start:.1f}s.")

    # ---------- Final Summary ----------
    final_text = st.session_state.get("final_text", "")
    st.subheader("Final Summary")
    if final_text and not st.session_state.get("final_displayed"):
        asyncio.run(stream_markdown_smooth(final_text, delay=0.1))
        st.session_state["final_displayed"] = True
    elif final_text:
        st.markdown(final_text)

    if final_text:
        st.download_button("Download Final Summary", final_text, "final_summary.txt")

    # ---------- Page-wise summaries ----------
    st.subheader("Page Summaries")
    page_links = []
    for txt_file in sorted(OUTPUT_DIR.glob("page_*.txt")):
        txt = txt_file.read_text(encoding="utf-8")
        html = make_data_link(f"Open {txt_file.name}", txt, txt_file.name)
        page_links.append(html)
    if page_links:
        st.markdown("<br>".join(page_links), unsafe_allow_html=True)

    # ---------- Chat Agent ----------
    if st.session_state.get("kb_ready"):
        st.subheader("💬 Chat with MedGuide")
        if "chat_agent_obj" not in st.session_state:
            ag = chat_agent(
                lancedb_path=str(LANCEDB_DIR),
                collection=TABLE_NAME,
                top_k=5,
                min_docs_for_confident_answer=1,
                use_reranker=bool(enable_reranker and cohere_key),
                db_path=str(DATA_DIR / "agent_memory.db"),
                enable_agentic_memory=False,
            )
            try:
                ag.model = OpenAIChat(id=model_id)
            except Exception:
                pass
            st.session_state["chat_agent_obj"] = ag
            st.session_state["messages"] = []
            st.session_state["session_id"] = "st-session-1"

        ag = st.session_state["chat_agent_obj"]
        for m in st.session_state["messages"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_msg = st.chat_input("Ask about your report, diet, or next steps...")
        if user_msg:
            st.session_state["messages"].append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ans, _ = ag.answer(user_msg, user_id="st-user", session_id="st-session-1")
                    st.markdown(ans)
            st.session_state["messages"].append({"role": "assistant", "content": ans})

else:
    st.info("Upload a PDF and click Process to begin.")
