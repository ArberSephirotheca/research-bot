# scripts/

Helper scripts for RAG ingestion.

## rag_ingest.py

Builds `rag/chunks.jsonl` and `rag/embeddings.jsonl` from PDF papers and report Markdown.

Dependencies:

```bash
pip install pypdf requests
```

Usage:

```bash
python scripts/rag_ingest.py --input-dir papers --reports-dir reports
```

The script reads `OPENAI_API_KEY` (and optional `OPENAI_EMBED_MODEL`, `OPENAI_BASE_URL`) from your environment or `.env`.
Omit `--reports-dir` if you only want PDFs.

## download_papers.py

Scans `reports/` for arXiv links and downloads PDFs into `papers/`.

```bash
python scripts/download_papers.py --reports-dir reports --out-dir papers
```

Use `--report` to target a specific report file.

## sync_rag_artifacts.sh

Downloads the latest `rag-artifacts` workflow output via `gh` CLI, syncs `papers/` and `rag/` into the repo, and restarts the Slack bot service.

This script is intended to be run via the systemd units under `scripts/systemd/`.
