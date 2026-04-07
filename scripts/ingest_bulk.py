#!/usr/bin/env python3
"""
Bulk ingest all PDFs from a folder into Pinecone.

Usage:
    python scripts/ingest_bulk.py --folder ./pdfs --namespace my-docs
"""
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.pdf_processor import PDFProcessor
from app.services.embedder import OllamaEmbedder
from app.services.vector_store import VectorStore

app = typer.Typer()
console = Console()


@app.command()
def ingest(
    folder: Path = typer.Option(..., "--folder", "-f", help="Folder containing PDFs"),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Pinecone namespace"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Search subfolders"),
):
    setup_logging()
    settings = get_settings()

    if not folder.exists():
        console.print(f"[red]Folder not found: {folder}[/red]")
        raise typer.Exit(1)

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(folder.glob(pattern))

    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {folder}[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[bold]RAG PDF Bulk Ingest[/bold]")
    console.print(f"  Found [cyan]{len(pdf_files)}[/cyan] PDF(s) in [cyan]{folder}[/cyan]")
    console.print(f"  Namespace: [cyan]{namespace}[/cyan]")
    console.print(f"  Index: [cyan]{settings.pinecone_index_name}[/cyan]\n")

    processor = PDFProcessor()
    embedder = OllamaEmbedder()
    vector_store = VectorStore()

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting PDFs...", total=len(pdf_files))

        for pdf_path in pdf_files:
            progress.update(task, description=f"Processing {pdf_path.name[:40]}...")
            try:
                content = pdf_path.read_bytes()
                doc = processor.process(content, pdf_path.name)
                texts = [c.text for c in doc.chunks]
                embeddings = embedder.embed_batch(texts)
                upserted = vector_store.upsert_chunks(doc.chunks, embeddings, namespace=namespace)
                results.append({
                    "file": pdf_path.name,
                    "pages": doc.pages,
                    "chunks": upserted,
                    "status": "ok",
                    "doc_id": doc.doc_id,
                })
            except Exception as e:
                results.append({
                    "file": pdf_path.name,
                    "pages": 0,
                    "chunks": 0,
                    "status": f"error: {e}",
                    "doc_id": "-",
                })
            progress.advance(task)

    # Summary table
    table = Table(title="Ingestion Results", show_lines=True)
    table.add_column("File", style="cyan", max_width=40)
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        status_style = "green" if r["status"] == "ok" else "red"
        table.add_row(
            r["file"],
            str(r["pages"]),
            str(r["chunks"]),
            f"[{status_style}]{r['status']}[/{status_style}]",
        )

    console.print(table)
    ok = sum(1 for r in results if r["status"] == "ok")
    console.print(f"\n[bold]Done:[/bold] {ok}/{len(results)} files indexed successfully.\n")


if __name__ == "__main__":
    app()