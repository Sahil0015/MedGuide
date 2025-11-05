from __future__ import annotations
from typing import Tuple, Optional
import os
import pathlib

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.reranker.cohere import CohereReranker
from agno.db.sqlite import SqliteDb


def chat_agent(
    lancedb_path: str,
    collection: str = "medguide_collection",
    top_k: int = 5,
    min_docs_for_confident_answer: int = 1,
    use_reranker: bool = True,
    db_path: str = "data/agent_memory.db",
    enable_agentic_memory: bool = False,
) -> Agent:
    """
    MedGuide conversational agent using LanceDB for vector search
    and DuckDuckGo for fallback web retrieval.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("❌ OPENAI_API_KEY not set.")

    # --- Ensure persistence paths exist ---
    lancedb_path = str(pathlib.Path(lancedb_path))
    db_path = str(pathlib.Path(db_path))
    pathlib.Path(lancedb_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # --- Memory DB for chat history ---
    db = SqliteDb(db_file=db_path)

    # --- Vector knowledge base (LanceDB) ---
    knowledge = Knowledge(
        vector_db=LanceDb(
            table_name=collection,              # CHANGED: was `collection`
            uri=lancedb_path,                   # CHANGED: was `path`
            search_type=SearchType.hybrid,      # NEW: enable hybrid retrieval
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
            reranker=CohereReranker(model="rerank-multilingual-v3.0") if use_reranker else None,
        )
    )

    # --- Model + tools ---
    model = OpenAIChat(id="gpt-4o-mini")
    web_tool = DuckDuckGoTools(enable_search=True)

    # --- Agent setup ---
    agent = Agent(
        name="chat_agent",
        model=model,
        knowledge=knowledge,
        tools=[web_tool],
        system_message=(
            "You are MedGuide. Use retrieved local reports and summaries first; "
            "if retrieval is weak, use web search to supplement before answering."
            "If no relevant retrieved information is found, do not make up answers or use web search;"
            "Only answer questions related to medical lab reports and health summaries."
            "Do not answer questions about anything else."
        ),
        db=db,
        enable_user_memories=not enable_agentic_memory,
        enable_agentic_memory=enable_agentic_memory,
        add_history_to_context=True,
        num_history_runs=3,
        read_chat_history=True,
    )

    # --- Helper ---
    def _text(o) -> str:
        return getattr(o, "content", str(o)) or ""

    # --- Custom answer pipeline ---
    def answer(
        query: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Tuple[str, int]:
        try:
            retrieved = knowledge.vector_db.search(query=query) or []
        except IndexError:
            print("⚠️ Empty LanceDB search result. Returning [].")
            retrieved = []
        except Exception as e:
            print(f"⚠️ LanceDB search failed: {e}")
            retrieved = []
        retrieved = retrieved[:top_k]
        count = len(retrieved)

        def _doc_text(d):
            return (
                getattr(d, "text", None)
                or getattr(d, "page_content", None)
                or getattr(d, "content", "")
                or ""
            )

        context = "\n\n".join([_doc_text(d) for d in retrieved if _doc_text(d)])

        if count < min_docs_for_confident_answer:
            prompt = f"""
                If the user query is not about medical lab results, reply only:
                "Sorry, I can only help with lab reports and medical test interpretation."

                The user asked a lab-related question: "{query}"

                First, use the local context below. If it seems insufficient, call the web search tool to fetch 1–2 authoritative lines
                (acceptable domains: nih.gov, mayoclinic.org, medlineplus.gov) and then answer concisely.

                Local context:
                {context}

                When answering:
                - Be concise, structured, and factual
                - Cite which snippets you relied on
                - Do not diagnose or prescribe
                """
            out = agent.run(prompt, user_id=user_id, session_id=session_id)
            return _text(out), count

        prompt = f"""
            If the user query is not about medical lab results, reply only:
            "Sorry, I can only help with lab reports and medical test interpretation."

            Use the retrieved local medical data to answer precisely.

            User query:
            {query}

            Retrieved context:
            {context}

            Provide a clear, factual, and concise answer based on the above.
            """
        out = agent.run(prompt, user_id=user_id, session_id=session_id)
        return _text(out), count


    agent.answer = answer
    return agent

