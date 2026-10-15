"""
streamlit_app.py
----------------
Interactive Streamlit Dashboard for the Self-Corrective RAG Technical Documentation Assistant.
Features:
1. 🔎 RAG Query & Pipeline Inspector: Real-time LangGraph node execution trace and grounded answers.
2. 📥 Data Ingestion & Corpus Management: Ingest files (.md, .txt, .pdf) or scrape URLs via FastAPI /ingest endpoints.
3. 🏥 API Health & Architecture: Real-time backend diagnostics and Mermaid graph topology.

Usage:
    & "d:\\EA_Assignment\\venv\\Scripts\\streamlit.exe" run streamlit_app.py
"""

import os
import sys
import uuid
import urllib.parse
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import requests

from rag_pipeline.graph import get_rag_graph
from api.services.feedback_store import save_feedback, get_feedback_stats
from api.services.ingestion_service import (
    list_indexed_documents,
    ingest_text_content,
    extract_text_from_url,
    extract_text_from_pdf,
    delete_document_by_name,
)

# ── Page Configuration & Global Styling ───────────────────────────────────────
st.set_page_config(
    page_title="Self-Corrective RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .endpoint-card {
        background: #0f172a;
        padding: 14px 20px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 1.2rem;
    }
    .endpoint-badge {
        background-color: #1e293b;
        color: #38bdf8;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.88rem;
        font-weight: 600;
        border: 1px solid #475569;
    }
    .feedback-card {
        background-color: #0f172a;
        padding: 18px 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State Initialization ──────────────────────────────────────────────
if "current_result" not in st.session_state:
    st.session_state["current_result"] = None
if "current_query_id" not in st.session_state:
    st.session_state["current_query_id"] = None
if "feedback_submitted" not in st.session_state:
    st.session_state["feedback_submitted"] = False
if "api_url" not in st.session_state:
    st.session_state["api_url"] = "http://localhost:8000"


# ── Helper API Client Functions ───────────────────────────────────────────────
def check_api_health(base_url: str) -> Optional[Dict[str, Any]]:
    """Checks whether the FastAPI backend is running and returns health json."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=1.5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def api_ingest_file(base_url: str, uploaded_file, doc_name: str) -> Dict[str, Any]:
    """Ingests file via POST /ingest endpoint."""
    url = f"{base_url.rstrip('/')}/ingest"
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
    data = {}
    if doc_name.strip():
        data["document_name"] = doc_name.strip()
    resp = requests.post(url, files=files, data=data, timeout=60)
    if resp.status_code != 201:
        raise RuntimeError(f"API Error ({resp.status_code}): {resp.text}")
    return resp.json()


def api_ingest_url(base_url: str, doc_url: str, doc_name: str) -> Dict[str, Any]:
    """Ingests documentation via POST /ingest/url endpoint."""
    url = f"{base_url.rstrip('/')}/ingest/url"
    payload = {"url": doc_url.strip()}
    if doc_name.strip():
        payload["document_name"] = doc_name.strip()
    resp = requests.post(url, json=payload, timeout=60)
    if resp.status_code != 201:
        raise RuntimeError(f"API Error ({resp.status_code}): {resp.text}")
    return resp.json()


def api_list_documents(base_url: str) -> Dict[str, Any]:
    """Fetches corpus catalog via GET /documents endpoint."""
    url = f"{base_url.rstrip('/')}/documents"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"API Error ({resp.status_code}): {resp.text}")
    return resp.json()


def api_delete_document(base_url: str, document_name: str) -> Dict[str, Any]:
    """Deletes document chunks via DELETE /documents/{name} endpoint."""
    encoded_name = urllib.parse.quote(document_name)
    url = f"{base_url.rstrip('/')}/documents/{encoded_name}"
    resp = requests.delete(url, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"API Error ({resp.status_code}): {resp.text}")
    return resp.json()


# ── Sidebar Navigation & Configuration ────────────────────────────────────────
st.sidebar.markdown("## 🧭 Application Navigation")
app_mode = st.sidebar.radio(
    "Choose View:",
    options=[
        "🔎 RAG Assistant & Inspector",
        "📥 Data Ingestion & Corpus Management",
        "🏥 API Health & Architecture",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Backend API Connection")
backend_url_input = st.sidebar.text_input(
    "FastAPI Base URL:",
    value=st.session_state["api_url"],
    help="Default FastAPI server URL (running on port 8000 via run_all.py or uvicorn)",
)
st.session_state["api_url"] = backend_url_input.strip()

# Check live backend health
health_data = check_api_health(st.session_state["api_url"])
api_is_online = health_data is not None

if api_is_online:
    st.sidebar.success(f"🟢 Backend Online (`port 8000`)")
else:
    st.sidebar.warning("🟡 Backend Offline (Direct Service Mode Active)")
    st.sidebar.caption("Run `python run_all.py` to start both FastAPI & Streamlit.")


# ==============================================================================
# VIEW 1: 📥 DATA INGESTION & CORPUS MANAGEMENT
# ==============================================================================
if app_mode == "📥 Data Ingestion & Corpus Management":
    st.markdown("<div class='main-header'>📥 Data Ingestion & Corpus Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload technical files or scrape documentation URLs into ChromaDB & BM25 indices via FastAPI endpoints</div>", unsafe_allow_html=True)

    # API Endpoint Explanation Card
    status_label = "🟢 Connected to FastAPI Server" if api_is_online else "🟡 Direct Python Service Fallback"
    st.markdown(
        f"""
        <div class='endpoint-card'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <strong>FastAPI Ingestion Pipeline:</strong>
                <span style='font-size:0.85rem; color:#94a3b8;'>Status: <b>{status_label}</b></span>
            </div>
            <p style='color:#cbd5e1; font-size:0.92rem; margin-top:6px; margin-bottom:6px;'>
                Endpoints used:
                <span class='endpoint-badge'>POST /ingest</span> (File & URL Multipart) &nbsp;|&nbsp;
                <span class='endpoint-badge'>POST /ingest/url</span> (JSON URL Crawler) &nbsp;|&nbsp;
                <span class='endpoint-badge'>GET /documents</span> (Catalog) &nbsp;|&nbsp;
                <span class='endpoint-badge'>DELETE /documents/{{name}}</span> (Chunk Removal)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_upload, tab_url, tab_catalog = st.tabs([
        "📄 File Upload Ingestion",
        "🌐 Web URL Documentation Scraping",
        "📚 Corpus Catalog & Management",
    ])

    # ── Tab 1: File Upload Ingestion ───────────────────────────────────────────
    with tab_upload:
        st.markdown("### 📄 Ingest Technical Document via File Upload")
        st.markdown("Upload markdown, plain text, or PDF technical guides. Documents will be chunked with markdown header awareness and indexed into vector + sparse BM25 storage.")

        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            uploaded_file = st.file_uploader(
                "Select documentation file:",
                type=["md", "txt", "pdf"],
                help="Supported file formats: Markdown (.md), Plain text (.txt), and PDF (.pdf)",
            )
        with col_f2:
            custom_doc_name = st.text_input(
                "Document Name (Optional):",
                placeholder="e.g., FastAPI Architecture Guide",
                help="Leave empty to automatically infer from the filename.",
            )

        ingest_file_btn = st.button("🚀 Ingest Document via API", type="primary", disabled=uploaded_file is None)

        if ingest_file_btn and uploaded_file is not None:
            with st.spinner(f"Ingesting '{uploaded_file.name}' into vector database..."):
                try:
                    if api_is_online:
                        # Call FastAPI POST /ingest
                        res = api_ingest_file(
                            base_url=st.session_state["api_url"],
                            uploaded_file=uploaded_file,
                            doc_name=custom_doc_name,
                        )
                        st.success(f"✅ **Ingestion Successful via FastAPI Endpoint (`POST /ingest`)!**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Document Name", res.get("document_name", uploaded_file.name))
                        c2.metric("Chunks Added", f"+{res.get('chunks_added', 0)}")
                        c3.metric("Total Corpus Chunks", res.get("total_corpus_chunks", "N/A"))
                        st.info(f"💬 Message: {res.get('message', '')}")
                    else:
                        # Fallback to direct service call
                        filename = uploaded_file.name
                        doc_title = custom_doc_name.strip() or filename.rsplit(".", 1)[0].replace("_", " ").title()
                        bytes_content = uploaded_file.getvalue()

                        if filename.lower().endswith(".pdf"):
                            text = extract_text_from_pdf(bytes_content)
                        else:
                            text = bytes_content.decode("utf-8", errors="replace")

                        chunks_added, total = ingest_text_content(
                            content=text,
                            source=filename,
                            document_name=doc_title,
                        )
                        st.success(f"✅ **Ingestion Successful (Direct Service Execution)!**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Document Name", doc_title)
                        c2.metric("Chunks Added", f"+{chunks_added}")
                        c3.metric("Total Corpus Chunks", total)
                        st.info(f"Indexed {chunks_added} chunks from '{filename}'.")

                except Exception as exc:
                    st.error(f"❌ Ingestion Failed: {str(exc)}")

    # ── Tab 2: Web URL Documentation Scraping ──────────────────────────────────
    with tab_url:
        st.markdown("### 🌐 Scrape and Ingest Public Documentation by URL")
        st.markdown("Extract live HTML documentation, parse sections, clean scripts and headers, and index the content into the RAG corpus.")

        col_u1, col_u2 = st.columns([3, 2])
        with col_u1:
            url_input = st.text_input(
                "Documentation Web Page URL:",
                placeholder="https://fastapi.tiangolo.com/tutorial/path-params/",
            )
        with col_u2:
            url_doc_name = st.text_input(
                "Document Name (Optional):",
                placeholder="e.g., FastAPI Path Parameters Tutorial",
                key="url_doc_name_input",
            )

        st.caption("Quick sample URLs:")
        col_p1, col_p2, col_p3 = st.columns(3)
        if col_p1.button("📌 FastAPI Path Params"):
            url_input = "https://fastapi.tiangolo.com/tutorial/path-params/"
        if col_p2.button("📌 LangGraph Quickstart"):
            url_input = "https://langchain-ai.github.io/langgraph/tutorials/introduction/"
        if col_p3.button("📌 Pydantic Field Validator"):
            url_input = "https://docs.pydantic.dev/latest/concepts/validators/"

        ingest_url_btn = st.button("🌐 Scrape & Ingest via API", type="primary", disabled=not bool(url_input and url_input.strip()))

        if ingest_url_btn and url_input.strip():
            target_url = url_input.strip()
            with st.spinner(f"Scraping and indexing content from {target_url}..."):
                try:
                    if api_is_online:
                        # Call FastAPI POST /ingest/url
                        res = api_ingest_url(
                            base_url=st.session_state["api_url"],
                            doc_url=target_url,
                            doc_name=url_doc_name,
                        )
                        st.success(f"✅ **URL Ingestion Successful via FastAPI Endpoint (`POST /ingest/url`)!**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Indexed Name", res.get("document_name", target_url))
                        c2.metric("Chunks Added", f"+{res.get('chunks_added', 0)}")
                        c3.metric("Total Corpus Chunks", res.get("total_corpus_chunks", "N/A"))
                        st.info(f"💬 Message: {res.get('message', '')}")
                    else:
                        # Direct service fallback
                        page_title, text = extract_text_from_url(target_url)
                        doc_title = url_doc_name.strip() or page_title or target_url
                        chunks_added, total = ingest_text_content(
                            content=text,
                            source=target_url,
                            document_name=doc_title,
                        )
                        st.success(f"✅ **URL Ingestion Successful (Direct Service Execution)!**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Indexed Name", doc_title)
                        c2.metric("Chunks Added", f"+{chunks_added}")
                        c3.metric("Total Corpus Chunks", total)
                        st.info(f"Indexed {chunks_added} chunks from '{target_url}'.")

                except Exception as exc:
                    st.error(f"❌ URL Ingestion Failed: {str(exc)}")

    # ── Tab 3: Corpus Catalog & Management ─────────────────────────────────────
    with tab_catalog:
        st.markdown("### 📚 Corpus Catalog & Chunk Management")
        st.markdown("Inspect all active documentation collections, view chunk allocations, and delete obsolete entries from ChromaDB & BM25 indices.")

        col_r1, col_r2 = st.columns([1, 4])
        reload_btn = col_r1.button("🔄 Refresh Catalog")

        try:
            if api_is_online:
                catalog_data = api_list_documents(st.session_state["api_url"])
                total_docs = catalog_data.get("total_documents", 0)
                total_chunks = catalog_data.get("total_chunks", 0)
                docs_list = catalog_data.get("documents", [])
            else:
                cat_resp = list_indexed_documents()
                total_docs = cat_resp.total_documents
                total_chunks = cat_resp.total_chunks
                docs_list = [
                    {
                        "document_name": d.document_name,
                        "source": d.source,
                        "chunk_count": d.chunk_count,
                        "sample_sections": d.sample_sections,
                    }
                    for d in cat_resp.documents
                ]

            # Top KPI metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Indexed Documents", total_docs)
            m2.metric("Total Chunks in ChromaDB", total_chunks)
            m3.metric("Hybrid Index Status", "ChromaDB + BM25 Ready")

            st.divider()

            if docs_list:
                # Search filter
                search_term = st.text_input("🔍 Filter catalog by document name or source:", "")
                filtered_docs = [
                    d for d in docs_list
                    if not search_term or (search_term.lower() in d["document_name"].lower() or search_term.lower() in d["source"].lower())
                ]

                # Tabular overview
                rows = []
                for i, d in enumerate(filtered_docs, 1):
                    sec_preview = ", ".join(d.get("sample_sections", [])) or "General"
                    rows.append({
                        "#": i,
                        "Document Name": d.get("document_name"),
                        "Source / Filename": d.get("source"),
                        "Chunk Count": d.get("chunk_count"),
                        "Sample Sections": sec_preview[:80] + ("..." if len(sec_preview) > 80 else ""),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # Delete document tool
                st.markdown("#### 🗑️ Delete Document from Corpus")
                doc_options = [d["document_name"] for d in docs_list]
                selected_del = st.selectbox("Select document to remove:", doc_options)

                if st.button("⚠️ Delete Selected Document", type="secondary"):
                    with st.spinner(f"Deleting '{selected_del}' from ChromaDB and rebuilding BM25 index..."):
                        try:
                            if api_is_online:
                                del_res = api_delete_document(st.session_state["api_url"], selected_del)
                                st.success(f"Deleted {del_res.get('chunks_deleted', 0)} chunks. Remaining corpus chunks: {del_res.get('remaining_chunks', 0)}")
                            else:
                                del_count, rem = delete_document_by_name(selected_del)
                                st.success(f"Deleted {del_count} chunks. Remaining corpus chunks: {rem}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Deletion failed: {e}")
            else:
                st.info("Corpus is currently empty. Use the File Upload or Web URL tab to ingest documents.")

        except Exception as exc:
            st.error(f"Failed to load corpus catalog: {exc}")


# ==============================================================================
# VIEW 2: 🏥 API HEALTH & ARCHITECTURE
# ==============================================================================
elif app_mode == "🏥 API Health & Architecture":
    st.markdown("<div class='main-header'>🏥 System Diagnostics & Topology</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Real-time health telemetry from FastAPI and LangGraph multi-node architecture diagram</div>", unsafe_allow_html=True)

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("### 🔌 FastAPI Backend Status (`GET /health`)")
        if health_data:
            st.success("✅ FastAPI Backend is Healthy and Active!")
            st.json(health_data)
        else:
            st.error("❌ FastAPI Backend is currently unreachable.")
            st.caption(f"Attempted connection to `{st.session_state['api_url']}/health`.")
            st.markdown(
                """
                **How to start the FastAPI server:**
                ```bash
                # In your terminal:
                & "d:\\EA_Assignment\\venv\\Scripts\\python.exe" run_all.py
                # Or start uvicorn directly:
                uvicorn api.main:app --port 8000 --reload
                ```
                """
            )

    with col_h2:
        st.markdown("### 🧬 LangGraph Architecture Workflow")
        st.markdown(
            """
            ```mermaid
            flowchart TD
                A([User Question]) --> B[analyze_query]
                B --> C[retrieve]
                C --> D[grade_documents]
                D -->|Any Document Relevant| E[generate]
                D -->|All Irrelevant & Retry < 2| F[rewrite_query]
                F --> C
                D -->|All Irrelevant & Retry >= 2| G[web_search]
                G --> E
                E --> H[check_hallucination]
                H -->|Grounded Answer| I([Final Grounded Answer])
                H -->|Hallucinated & Regen < 2| E
                H -->|Regen Max Reached| I
            ```
            """
        )


# ==============================================================================
# VIEW 3: 🔎 RAG ASSISTANT & PIPELINE INSPECTOR
# ==============================================================================
else:
    # ── Sidebar for RAG View ──────────────────────────────────────────────────
    st.sidebar.title("⚙️ RAG Pipeline Config")
    st.sidebar.markdown(
        """
        **Architecture:**
        - **Workflow:** LangGraph StateGraph
        - **Local Hybrid:** ChromaDB (Dense) + BM25 (Sparse) via RRF ($k=60$)
        - **Grader:** Groq LLM Relevance Grader
        - **Web Fallback:** Tavily (Keyword) + Exa (Semantic) via RRF
        - **Auditor:** Self-RAG Hallucination Check
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Preset Queries")

    preset_options = [
        "-- Choose a preset query --",
        "How do I define query parameters and path parameters in FastAPI?",
        "How do conditional edges work in LangGraph and what do they return?",
        "How do I use field_validator in Pydantic V2?",
        "How do I filter query results by metadata in ChromaDB?",
        "[Web Fallback] What is the newest feature in Python 3.13 free threading?",
        "[Web Fallback] What are the latest open source models available on Groq in 2026?",
    ]

    selected_preset = st.sidebar.selectbox("Click to auto-populate:", preset_options)

    # ── Sidebar Feedback Analytics ────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 User Feedback Analytics")
    try:
        stats = get_feedback_stats()
        col_sb1, col_sb2 = st.sidebar.columns(2)
        col_sb1.metric("Satisfaction", f"{stats.satisfaction_rate_percent}%")
        col_sb2.metric("Total Ratings", stats.total_feedback)
        st.sidebar.caption(f"👍 {stats.thumbs_up} Thumbs Up &nbsp;|&nbsp; 👎 {stats.thumbs_down} Thumbs Down")

        if stats.feedback_entries:
            with st.sidebar.expander("Recent Feedback Log"):
                for entry in stats.feedback_entries[:4]:
                    icon = "👍" if entry.rating == "thumbs_up" else "👎"
                    st.markdown(f"**{icon} {entry.rating.replace('_', ' ').title()}**")
                    if entry.comment:
                        st.caption(f'"{entry.comment}"')
                    st.caption(f"🕒 {entry.created_at[:19].replace('T', ' ')}")
                    st.divider()
    except Exception as e:
        st.sidebar.caption(f"Feedback stats unavailable: {e}")

    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Clear Conversation & State"):
        st.session_state["current_result"] = None
        st.session_state["current_query_id"] = None
        st.session_state["feedback_submitted"] = False
        st.rerun()

    # ── Main UI ───────────────────────────────────────────────────────────────
    st.markdown("<div class='main-header'>Self-Corrective RAG Documentation Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Interactive End-to-End Pipeline Inspector with Multi-Node LangGraph Execution</div>", unsafe_allow_html=True)

    default_query = ""
    if selected_preset and selected_preset != "-- Choose a preset query --":
        default_query = selected_preset.replace("[Web Fallback] ", "")

    user_query = st.text_input(
        "Enter technical question:",
        value=default_query,
        placeholder="e.g., How do conditional edges work in LangGraph?",
    )

    col_run, col_info = st.columns([1, 5])
    run_button = col_run.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    if run_button and user_query.strip():
        question = user_query.strip()
        query_id = str(uuid.uuid4())

        initial_state = {
            "question": question,
            "query_intent": None,
            "rewritten_query": None,
            "documents": [],
            "filtered_documents": [],
            "document_grades": [],
            "generation": None,
            "retry_count": 0,
            "web_search_needed": False,
            "web_search_used": False,
            "is_grounded": None,
            "hallucination_explanation": None,
            "regen_count": 0,
        }

        graph = get_rag_graph()
        final_state = initial_state.copy()

        status_box = st.status("🔄 Executing Self-Corrective LangGraph Pipeline...", expanded=True)

        trace_events = []
        retrieved_docs_history = []
        web_docs_history = []
        rewrite_history = []

        for event in graph.stream(initial_state):
            for node_name, node_output in event.items():
                final_state.update(node_output)
                trace_events.append((node_name, node_output))

                if node_name == "analyze_query":
                    intent = node_output.get("query_intent", "general")
                    expanded = node_output.get("rewritten_query", "")
                    status_box.write(f"🔍 **[Query Analysis]** Classified intent as `{intent}`. Expanded query: *'{expanded}'*")

                elif node_name == "retrieve":
                    docs = node_output.get("documents", [])
                    retrieved_docs_history.append(docs)
                    status_box.write(f"📚 **[Hybrid Retrieval]** Fused Dense + BM25 via RRF. Retrieved {len(docs)} chunks from ChromaDB.")

                elif node_name == "grade_documents":
                    filtered = node_output.get("filtered_documents", [])
                    grades = node_output.get("document_grades", [])
                    rel_count = len(filtered)
                    total = len(grades) if grades else len(filtered)
                    if rel_count > 0:
                        status_box.write(f"⚖️ **[Document Grading]** {rel_count}/{total} chunks graded as **RELEVANT**.")
                    else:
                        status_box.write(f"⚠️ **[Document Grading]** All {total} chunks graded as **IRRELEVANT**.")

                elif node_name == "rewrite_query":
                    retry = node_output.get("retry_count", 0)
                    rewritten = node_output.get("rewritten_query", "")
                    rewrite_history.append((retry, rewritten))
                    status_box.write(f"🔁 **[Self-Correction Loop]** (Retry {retry}/2): Reformulating query to *'{rewritten}'*")

                elif node_name == "web_search":
                    web_docs = node_output.get("documents", [])
                    web_docs_history = web_docs
                    status_box.write(f"🌐 **[Web Search Fallback]** Dispatched parallel Tavily (Keyword) + Exa (Semantic) RRF! Received {len(web_docs)} web documents.")

                elif node_name == "generate":
                    status_box.write("💡 **[Generation]** Synthesizing grounded technical answer from validated context...")

                elif node_name == "check_hallucination":
                    is_grounded = node_output.get("is_grounded")
                    if is_grounded:
                        status_box.write("🛡️ **[Hallucination Audit]** Grounding Check: **PASSED (GROUNDED)**.")
                    else:
                        status_box.write("⚠️ **[Hallucination Audit]** Grounding Check: **REGENERATING**.")

        status_box.update(label="✅ Pipeline Execution Complete!", state="complete", expanded=False)

        # Persist in session state for feedback submission
        st.session_state["current_result"] = {
            "final_state": final_state,
            "retrieved_docs_history": retrieved_docs_history,
            "web_docs_history": web_docs_history,
            "rewrite_history": rewrite_history,
        }
        st.session_state["current_query_id"] = query_id
        st.session_state["feedback_submitted"] = False

    # ── Render Results from Session State ─────────────────────────────────────
    if st.session_state.get("current_result"):
        res = st.session_state["current_result"]
        final_state = res["final_state"]
        retrieved_docs_history = res["retrieved_docs_history"]
        web_docs_history = res["web_docs_history"]
        rewrite_history = res["rewrite_history"]

        st.markdown("### 🧩 Step-by-Step Pipeline Inspection")

        tab_nodes, tab_retrieval, tab_web, tab_audit = st.tabs([
            "Execution Summary", "Retrieved Local Chunks", "Web Search Fallback", "Hallucination Audit"
        ])

        with tab_nodes:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Query Intent", final_state.get("query_intent", "N/A").upper())
            col_m2.metric("Retries / Rewrites", f"{final_state.get('retry_count', 0)} / 2")
            col_m3.metric("Web Search Used", "YES" if final_state.get("web_search_used") else "NO")
            col_m4.metric("Hallucination Audit", "GROUNDED" if final_state.get("is_grounded") else "N/A")

            if rewrite_history:
                st.markdown("**Query Reformulations Triggered:**")
                for r_num, r_q in rewrite_history:
                    st.info(f"Attempt {r_num}/2: {r_q}")

        with tab_retrieval:
            if retrieved_docs_history:
                latest_docs = retrieved_docs_history[-1]
                rows = []
                for idx, d in enumerate(latest_docs, 1):
                    rows.append({
                        "#": idx,
                        "Source": d.metadata.get("source", "unknown"),
                        "Section": d.metadata.get("section", "General"),
                        "RRF Score": f"{d.metadata.get('rrf_score', 0.0):.5f}",
                        "Snippet": d.page_content[:150] + "...",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.write("No local documents retrieved.")

        with tab_web:
            if final_state.get("web_search_used") and web_docs_history:
                st.success("Web Search Fallback was active for this query!")
                web_rows = []
                for idx, d in enumerate(web_docs_history, 1):
                    web_rows.append({
                        "#": idx,
                        "Provider": d.metadata.get("search_provider", "web").upper(),
                        "RRF Score": f"{d.metadata.get('rrf_score', 0.0):.5f}",
                        "URL": d.metadata.get("source", ""),
                        "Title": d.metadata.get("title", ""),
                        "Preview": d.page_content[:140] + "...",
                    })
                st.dataframe(pd.DataFrame(web_rows), use_container_width=True)
            else:
                st.info("Web search fallback was not needed (Local documentation answered this query).")

        with tab_audit:
            audit_exp = final_state.get("hallucination_explanation", "No audit explanation available.")
            if final_state.get("is_grounded"):
                st.success(f"**Audit Verdict: GROUNDED**\n\n{audit_exp}")
            else:
                st.warning(f"**Audit Verdict: UNVERIFIED**\n\n{audit_exp}")

        # ── Final Answer & Citations ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💡 Final Generated Answer")
        generation_text = final_state.get("generation", "No generation produced.")
        st.markdown(generation_text)

        # Citations
        used_docs = final_state.get("filtered_documents", []) or final_state.get("documents", [])
        seen = set()
        citations = []
        for d in used_docs:
            src = d.metadata.get("source", "unknown")
            sec = d.metadata.get("section", d.metadata.get("title", ""))
            key = (src, sec)
            if key not in seen:
                seen.add(key)
                citations.append((src, sec))

        if citations:
            st.markdown("#### 📚 Citations & Sources")
            for i, (src, sec) in enumerate(citations, 1):
                if sec and sec != "General":
                    st.markdown(f"- **[{src}]** - *{sec}*")
                else:
                    st.markdown(f"- **[{src}]**")

        # ── Interactive Feedback Collection Widget ────────────────────────────
        st.markdown("---")
        st.markdown("### 💬 Answer Feedback & Evaluation")

        current_qid = st.session_state.get("current_query_id", "unknown-query")

        if st.session_state.get("feedback_submitted"):
            st.success("🎉 Thank you! Your feedback has been recorded and added to the quality audit log.")
        else:
            st.markdown(f"Query Reference ID: `{current_qid}`")
            col_fb1, col_fb2 = st.columns([1, 2])

            with col_fb1:
                rating_choice = st.radio(
                    "Was this response helpful?",
                    options=["👍 Helpful (Thumbs Up)", "👎 Inaccurate / Hallucinated (Thumbs Down)"],
                    index=0,
                    key="feedback_radio",
                )

            with col_fb2:
                comment_input = st.text_area(
                    "Optional comments or corrections:",
                    placeholder="e.g. Excellent explanation, precise code snippet and clear source citation.",
                    key="feedback_comment",
                    height=95,
                )

            if st.button("📨 Submit Feedback", type="secondary"):
                rating_str = "thumbs_up" if "👍" in rating_choice else "thumbs_down"
                save_feedback(
                    query_id=current_qid,
                    rating=rating_str,
                    comment=comment_input.strip() or None,
                )
                st.session_state["feedback_submitted"] = True
                st.rerun()
