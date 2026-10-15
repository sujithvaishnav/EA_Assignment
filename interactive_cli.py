"""
interactive_cli.py
------------------
Interactive CLI runner for the Self-Corrective RAG Pipeline with real-time
LangGraph node execution tracking, inspection tables, and preset queries.

Usage:
    & "d:\\EA_Assignment\\venv\\Scripts\\python.exe" interactive_cli.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

from rag_pipeline.graph import get_rag_graph

console = Console()

PRESET_QUERIES = [
    ("FastAPI Path & Query Parameters", "How do I define query parameters and path parameters in FastAPI?"),
    ("LangGraph Conditional Edges", "How do conditional edges work in LangGraph and what do they return?"),
    ("Pydantic V2 Field Validators", "How do I use field_validator in Pydantic V2?"),
    ("ChromaDB Metadata Filtering", "How do I filter query results by metadata in ChromaDB?"),
    ("Web Fallback: Python 3.13 Free Threading", "What is the newest feature in Python 3.13 free threading?"),
    ("Web Fallback: Latest Groq Models 2026", "What are the latest open source models available on Groq in 2026?"),
]


def display_welcome():
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Self-Corrective RAG Technical Documentation Assistant[/bold cyan]\n"
            "[dim]LangGraph - Hybrid Retrieval (Dense+BM25) - Document Grading - Query Rewriting - Web Search Fallback (Tavily+Exa) - Hallucination Audit[/dim]",
            border_style="cyan",
            title="[bold yellow]Interactive Pipeline Inspector[/bold yellow]",
        )
    )


def run_interactive_query(question: str):
    console.print(f"\n[bold yellow]User Question:[/bold yellow] [bold white]{question}[/bold white]\n")

    graph = get_rag_graph()
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

    final_state = initial_state.copy()

    with console.status("[bold green]Executing LangGraph pipeline...[/bold green]", spinner="dots"):
        for event in graph.stream(initial_state):
            for node_name, node_output in event.items():
                final_state.update(node_output)

                if node_name == "analyze_query":
                    intent = node_output.get("query_intent", "unknown")
                    rewritten = node_output.get("rewritten_query", "")
                    console.print(
                        f"  [cyan][Analyze Query][/cyan] Intent: [bold]{intent}[/bold] | Expanded: '{rewritten}'"
                    )

                elif node_name == "retrieve":
                    docs = node_output.get("documents", [])
                    table = Table(title="[Retrieval Chunks: Dense + BM25 + RRF]", show_header=True, header_style="bold magenta")
                    table.add_column("#", style="dim", width=4)
                    table.add_column("Source / File", style="cyan", width=26)
                    table.add_column("Section", style="white", width=36)
                    table.add_column("RRF Score", justify="right", style="green", width=12)

                    for idx, d in enumerate(docs[:5], 1):
                        source = d.metadata.get("source", "unknown")
                        section = d.metadata.get("section", "General")[:34]
                        rrf = d.metadata.get("rrf_score", 0.0)
                        table.add_row(str(idx), source, section, f"{rrf:.5f}")
                    console.print(table)

                elif node_name == "grade_documents":
                    grades = node_output.get("document_grades", [])
                    filtered = node_output.get("filtered_documents", [])
                    rel_count = len(filtered)
                    total_count = len(grades) if grades else len(filtered)
                    style = "green" if rel_count > 0 else "bold red"
                    console.print(f"  [yellow][Grade Documents][/yellow] Relevant Chunks: [{style}]{rel_count}/{total_count}[/{style}]")

                elif node_name == "rewrite_query":
                    retry = node_output.get("retry_count", 0)
                    rewritten = node_output.get("rewritten_query", "")
                    console.print(
                        f"  [bold red][Self-Correction][/bold red] Retry [bold]{retry}/2[/bold] -> Reformulated Query: '{rewritten}'"
                    )

                elif node_name == "web_search":
                    docs = node_output.get("documents", [])
                    console.print(
                        f"  [bold blue][Web Search Fallback][/bold blue] Triggered Tavily (Keyword) + Exa (Semantic) RRF! Injected {len(docs)} web docs."
                    )
                    table = Table(title="[Web Search RRF Fused Top Results]", show_header=True, header_style="bold blue")
                    table.add_column("#", style="dim", width=4)
                    table.add_column("Provider", style="magenta", width=10)
                    table.add_column("RRF Score", justify="right", style="green", width=12)
                    table.add_column("URL", style="cyan")

                    for idx, d in enumerate(docs[:5], 1):
                        provider = d.metadata.get("search_provider", "web").upper()
                        rrf = d.metadata.get("rrf_score", 0.0)
                        url = d.metadata.get("source", "")
                        table.add_row(str(idx), provider, f"{rrf:.5f}", url)
                    console.print(table)

                elif node_name == "generate":
                    console.print("  [green][Generate][/green] Synthesizing grounded answer...")

                elif node_name == "check_hallucination":
                    is_grounded = node_output.get("is_grounded")
                    status_text = "[bold green]PASS (GROUNDED)[/bold green]" if is_grounded else "[bold red]FAIL (HALLUCINATION DETECTED)[/bold red]"
                    console.print(f"  [magenta][Hallucination Audit][/magenta] Grounding: {status_text}")
                    reason = node_output.get("hallucination_explanation", "")
                    if reason:
                        console.print(f"    [dim]{reason}[/dim]")

                elif node_name == "no_answer":
                    console.print("  [bold red][No Answer Fallback][/bold red] Fallback triggered.")

    generation = final_state.get("generation", "No generation produced.")
    filtered_docs = final_state.get("filtered_documents", []) or final_state.get("documents", [])

    console.print()
    console.print(Panel(Markdown(generation), title="[bold green]Final Answer[/bold green]", border_style="green"))

    citations = []
    seen = set()
    for doc in filtered_docs:
        src = doc.metadata.get("source", "unknown")
        sec = doc.metadata.get("section", doc.metadata.get("title", ""))
        key = (src, sec)
        if key not in seen:
            seen.add(key)
            citations.append((src, sec))

    if citations:
        console.print("[bold yellow]Citations & Referenced Sources:[/bold yellow]")
        for i, (src, sec) in enumerate(citations[:5], 1):
            if sec and sec != "General":
                console.print(f"  {i}. [cyan]{src}[/cyan] - [dim]{sec}[/dim]")
            else:
                console.print(f"  {i}. [cyan]{src}[/cyan]")
    console.print()


def main():
    display_welcome()

    while True:
        console.print("[bold cyan]Choose an option:[/bold cyan]")
        for i, (title, q) in enumerate(PRESET_QUERIES, 1):
            console.print(f"  [bold green]{i}.[/bold green] {title} [dim]({q[:55]}...)[/dim]")
        console.print(f"  [bold green]{len(PRESET_QUERIES) + 1}.[/bold green] [bold yellow]Enter Custom Query[/bold yellow]")
        console.print("  [bold red]0.[/bold red] Exit\n")

        choice = Prompt.ask("Select option", default=str(len(PRESET_QUERIES) + 1))

        if choice == "0":
            console.print("[bold green]Goodbye![/bold green]")
            break

        if choice.isdigit() and 1 <= int(choice) <= len(PRESET_QUERIES):
            selected_title, query = PRESET_QUERIES[int(choice) - 1]
            run_interactive_query(query)
        elif choice == str(len(PRESET_QUERIES) + 1):
            custom_q = Prompt.ask("\n[bold yellow]Type your question[/bold yellow]")
            if custom_q.strip():
                run_interactive_query(custom_q.strip())
        else:
            console.print("[red]Invalid choice. Please select from the menu.[/red]\n")


if __name__ == "__main__":
    main()
