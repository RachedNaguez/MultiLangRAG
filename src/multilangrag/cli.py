"""CLI interface for MultiLangRAG — polished Rich UI."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .config import RAGConfig
from .rag import build_chain

THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "dim": "dim",
        "accent": "bold bright_blue",
    }
)

console = Console(theme=THEME)

HISTORY: list[dict[str, str]] = []
QUERY_COUNT = 0
START_TIME: float = 0.0


def _uptime() -> str:
    elapsed = time.time() - START_TIME
    mins, secs = divmod(int(elapsed), 60)
    return f"{mins}m {secs}s" if mins else f"{secs}s"


def _print_banner() -> None:
    os.system("cls" if os.name == "nt" else "clear")

    title = Text()
    title.append("  M", style="bold cyan")
    title.append("ulti", style="bold bright_blue")
    title.append("Lang", style="bold magenta")
    title.append("RAG", style="bold bright_magenta")
    title.append("  ", style="")

    subtitle = Text("Multilingual Retrieval-Augmented Generation", style="dim")

    header = Table.grid(padding=(0, 1))
    header.add_row(title, subtitle)

    console.print()
    console.print(
        Panel(
            header,
            border_style="bright_blue",
            padding=(0, 1),
            width=62,
        )
    )

    hints = Table.grid(padding=(0, 1))
    hints.add_column(style="dim", width=3)
    hints.add_column()
    hints.add_row(">", "Ask questions in any language")
    hints.add_row(">", "[accent]/history[/accent]  past queries")
    hints.add_row(">", "[accent]/sources[/accent]  last retrieved documents")
    hints.add_row(">", "[accent]/clear[/accent]     clear screen")
    hints.add_row(">", "[accent]/quit[/accent]      exit")

    console.print(hints)
    console.print()
    console.print(
        f"  [dim]Session started {datetime.now().strftime('%H:%M:%S')} "
        f"| Model: [accent]{RAGConfig().llm_model}[/accent][/dim]"
    )
    console.print()


def _show_sources(docs) -> None:
    if not docs:
        console.print("\n  [dim]No sources retrieved yet.[/dim]\n")
        return

    table = Table(
        title="Retrieved Sources",
        show_lines=True,
        border_style="bright_blue",
        title_style="bold bright_blue",
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Language", style="cyan", width=10)
    table.add_column("Source", style="yellow", width=18)
    table.add_column("Content", style="white", ratio=3)

    for i, doc in enumerate(docs, 1):
        lang = doc.metadata.get("language", "?")
        source = doc.metadata.get("source", "?")
        content = doc.page_content[:100] + (
            "..." if len(doc.page_content) > 100 else ""
        )
        table.add_row(str(i), lang, source, content)

    console.print()
    console.print(table)
    console.print()


def _show_answer(answer: str, elapsed: float, docs=None) -> None:
    global QUERY_COUNT
    QUERY_COUNT += 1

    console.print()

    if docs:
        source_langs = ", ".join(
            sorted({d.metadata.get("language", "?") for d in docs})
        )
        source_label = f"Answer  [dim]({len(docs)} sources: {source_langs})[/dim]"
    else:
        source_label = "Answer"

    with Live(
        Panel(
            Markdown(answer),
            title=f"[bold green]{source_label}[/bold green]",
            border_style="green",
            padding=(0, 1),
            width=64,
        ),
        console=console,
        refresh_per_second=12,
    ):
        time.sleep(0.15)

    console.print(
        f"  [dim]({elapsed:.1f}s) #{QUERY_COUNT}[/dim]"
    )
    console.print()


def _show_error(exc: Exception) -> None:
    console.print()
    console.print(
        Panel(
            f"[error]{type(exc).__name__}:[/error] {exc}",
            title="[bold red]Error[/bold red]",
            border_style="red",
            padding=(0, 1),
            width=64,
        )
    )
    console.print()


def _show_history() -> None:
    if not HISTORY:
        console.print("\n  [dim]No history yet.[/dim]\n")
        return

    table = Table(
        title=f"Query History  [dim]({len(HISTORY)} total)[/dim]",
        show_lines=True,
        border_style="bright_blue",
        title_style="bold bright_blue",
        width=70,
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Question", style="cyan", ratio=2)
    table.add_column("Answer", style="white", ratio=3, max_width=35)

    for i, entry in enumerate(HISTORY, 1):
        q = entry["question"]
        a = entry["answer"]
        preview = a[:80] + ("..." if len(a) > 80 else "")
        table.add_row(str(i), q, preview)

    console.print()
    console.print(table)
    console.print()


def _show_help() -> None:
    table = Table(
        title="Commands",
        border_style="bright_blue",
        title_style="bold bright_blue",
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Command", style="bold yellow", no_wrap=True)
    table.add_column("Description")

    table.add_row("/history", "Show all past queries and answers")
    table.add_row("/sources", "Show documents retrieved in last query")
    table.add_row("/clear", "Clear the screen")
    table.add_row("/help", "Show this help message")
    table.add_row("/quit", "Exit the application")
    table.add_row("Ctrl+C", "Exit the application")

    console.print()
    console.print(table)
    console.print()


def _show_status() -> None:
    config = RAGConfig()
    table = Table(
        title="Session Status",
        border_style="bright_blue",
        title_style="bold bright_blue",
        padding=(0, 2),
        width=50,
    )
    table.add_column("Key", style="dim", width=14)
    table.add_column("Value", style="white")
    table.add_row("Model", f"[accent]{config.llm_model}[/accent]")
    table.add_row("Temperature", str(config.temperature))
    table.add_row("Top-K", str(config.top_k))
    table.add_row("Queries", str(QUERY_COUNT))
    table.add_row("Uptime", _uptime())
    console.print()
    console.print(table)
    console.print()


def _process_query(chain, retriever, query: str) -> list | None:
    start = time.perf_counter()

    with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
        try:
            docs = retriever.invoke(query)
            answer = chain.invoke(query)
        except Exception as exc:
            _show_error(exc)
            return None

    elapsed = time.perf_counter() - start
    HISTORY.append({"question": query, "answer": answer})
    _show_answer(answer, elapsed, docs)
    return docs


def interactive(config: RAGConfig | None = None) -> None:
    """Launch the interactive RAG chat."""
    global START_TIME

    if config is None:
        config = RAGConfig()

    START_TIME = time.time()
    _print_banner()

    last_docs = None

    with console.status("[cyan]Loading models and vector store...[/cyan]", spinner="dots"):
        pipeline = build_chain(config)
    chain = pipeline["chain"]
    retriever = pipeline["retriever"]

    console.print("  [success]Ready![/success]\n")

    while True:
        try:
            query = Prompt.ask("  [accent]>[/accent]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]Goodbye![/dim]")
            break

        query = query.strip()
        if not query:
            continue

        cmd = query.lower()

        if cmd in {"/quit", "/exit", "q"}:
            console.print("  [dim]Goodbye![/dim]")
            break

        if cmd == "/history":
            _show_history()
            continue

        if cmd == "/sources":
            _show_sources(last_docs)
            continue

        if cmd == "/clear":
            _print_banner()
            continue

        if cmd == "/help":
            _show_help()
            continue

        if cmd == "/status":
            _show_status()
            continue

        last_docs = _process_query(chain, retriever, query)


def query_once(question: str, config: RAGConfig | None = None) -> str:
    """Run a single query and return the answer."""
    if config is None:
        config = RAGConfig()
    pipeline = build_chain(config)
    return pipeline["chain"].invoke(question)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MultiLangRAG — Multilingual RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py                                   # interactive mode\n"
            '  python main.py "How do I handle IT issues?"       # single query\n'
            "  python main.py --model llama-3.1-8b-instant      # use different model\n"
        ),
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Ask a single question (non-interactive mode)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of documents to retrieve (default: 2)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature (default: 0.0)",
    )
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="ChromaDB persistence directory (default: data/chroma)",
    )
    args = parser.parse_args()

    config_kwargs = {}
    if args.model:
        config_kwargs["llm_model"] = args.model
    if args.top_k is not None:
        config_kwargs["top_k"] = args.top_k
    if args.temperature is not None:
        config_kwargs["temperature"] = args.temperature
    if args.persist_dir:
        config_kwargs["persist_directory"] = args.persist_dir

    config = RAGConfig(**config_kwargs)

    if args.question:
        with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
            answer = query_once(args.question, config)
        console.print(answer)
    else:
        interactive(config)


if __name__ == "__main__":
    main()
