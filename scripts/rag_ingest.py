#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from typing import Iterable, List

import requests
from pypdf import PdfReader


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def load_env_file(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value


def list_pdfs(root: str) -> List[str]:
    files = []
    for base, _, names in os.walk(root):
        for name in names:
            if name.lower().endswith(".pdf"):
                files.append(os.path.join(base, name))
    files.sort()
    return files


def list_markdown(root: str) -> List[str]:
    files = []
    for base, _, names in os.walk(root):
        for name in names:
            if not name.lower().endswith(".md"):
                continue
            if name.lower() == "readme.md":
                continue
            files.append(os.path.join(base, name))
    files.sort()
    return files


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def extract_markdown_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?;])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def split_long(text: str, max_chars: int) -> List[str]:
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = split_sentences(text)
    if not sentences:
        return split_long(text, max_chars)

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long(sentence, max_chars))
            continue
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current)
            if overlap > 0:
                current = current[-overlap:]
            else:
                current = ""
        current += sentence + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def iter_batches(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def post_with_retry(url: str, headers: dict, payload: dict) -> dict:
    for attempt in range(6):
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code in {429, 500, 502, 503, 504}:
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}


def embed_chunks(
    chunks: List[dict],
    api_key: str,
    model: str,
    base_url: str,
    batch_size: int,
) -> List[dict]:
    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    output = []
    for batch in iter_batches(chunks, batch_size):
        texts = [item["text"] for item in batch]
        payload = {"model": model, "input": texts}
        data = post_with_retry(url, headers, payload)
        embeddings = [row["embedding"] for row in data.get("data", [])]
        if len(embeddings) != len(batch):
            raise RuntimeError("Embedding response size mismatch")
        for item, emb in zip(batch, embeddings):
            enriched = dict(item)
            enriched["embedding"] = emb
            output.append(enriched)
    return output


def main() -> None:
    root = repo_root()
    load_env_file(os.path.join(root, ".env"))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing PDF papers (can be empty if using --reports-dir).",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Optional directory containing report Markdown to ingest.",
    )
    parser.add_argument("--output-dir", default="rag", help="Output directory for chunks/embeddings.")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Max chars per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--openai-embed-model", default=None, help="Override embedding model.")
    parser.add_argument("--openai-base-url", default=None, help="Override base URL.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    model = args.openai_embed_model or os.environ.get(
        "OPENAI_EMBED_MODEL", "text-embedding-3-small"
    )
    base_url = args.openai_base_url or os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    )

    input_dir = os.path.abspath(args.input_dir)
    pdfs = list_pdfs(input_dir)
    report_paths: List[str] = []
    if args.reports_dir:
        reports_dir = os.path.abspath(args.reports_dir)
        report_paths = list_markdown(reports_dir)
    if not pdfs and not report_paths:
        if args.reports_dir:
            raise RuntimeError(
                f"No PDFs found in {input_dir} and no reports found in {reports_dir}"
            )
        raise RuntimeError(
            f"No PDFs found in {input_dir} and no reports directory provided"
        )

    chunks: List[dict] = []
    for path in pdfs:
        text = extract_pdf_text(path)
        if not text.strip():
            raise RuntimeError(f"No text extracted from {path}")
        rel_path = os.path.relpath(path, root)
        for idx, chunk in enumerate(chunk_text(text, args.chunk_size, args.chunk_overlap)):
            chunks.append(
                {
                    "text": chunk,
                    "source": rel_path,
                    "chunk_index": idx,
                }
            )
    for path in report_paths:
        text = extract_markdown_text(path)
        if not text.strip():
            raise RuntimeError(f"No text extracted from {path}")
        rel_path = os.path.relpath(path, root)
        for idx, chunk in enumerate(chunk_text(text, args.chunk_size, args.chunk_overlap)):
            chunks.append(
                {
                    "text": chunk,
                    "source": rel_path,
                    "chunk_index": idx,
                }
            )

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    chunks_path = os.path.join(output_dir, "chunks.jsonl")
    embeddings_path = os.path.join(output_dir, "embeddings.jsonl")

    with open(chunks_path, "w", encoding="utf-8") as handle:
        for row in chunks:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    embedded = embed_chunks(chunks, api_key, model, base_url, args.batch_size)
    with open(embeddings_path, "w", encoding="utf-8") as handle:
        for row in embedded:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks to {chunks_path}")
    print(f"Wrote {len(embedded)} embeddings to {embeddings_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
